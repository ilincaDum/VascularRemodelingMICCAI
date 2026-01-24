"""
CSV-only LR/RF baselines aligned with the GNN framework

- Same logic and optional patient inclusion criteria as the GNN scripts
- LOPO-style folds (train/val/test by patient)
- No topology features used, only tabular version of the data

"""

import os, re, time, random, warnings, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(r"C:\Users\ilinc\OneDrive\Desktop\GraphAnalysis\GraphsCompleteAnalysis\graphs_complete_cleaned")
FLG  = ROOT / "final_labeled_graphs"
DAMAGE_ROOT = FLG / "damage_thresholds"

MODALITIES = ["Artery", "Vein"]
VALID_MODALITIES = ("Artery", "Vein")
THRESHOLDS = ["damage_90pct", "damage_75pct", "damage_50pct"]

SEED = 42

HARD_EXCLUDE_PATIENTS = {
    "P144","P151","P152","P155","P168","P172","P181","P183","P190","P109"
}

ENABLE_QC = True
FORCE_KEEP_ALL_PAIRS = False
MATCH_FRAC_THRESHOLD = 0.75
BAD_GRAPHS_PATIENT_EXCLUDE = 2
REPORT_CSV = FLG / "_postmatch_reports" / "lobe_counts_and_quality.csv"

TEST_GROUP_SIZE = 1
N_VAL_PATIENTS = 3

MAX_EDGES_PER_PAIR = 20000
MAX_EDGES_PER_PATIENT = 150000

SUBSAMPLE_STRATIFIED = True
SUBSAMPLE_KEEP_POS_FRACTION = 0.50

ADD_FU_TIME = True

TP_RE = re.compile(
    r"FU\s*0*(\d+)"
    r"(?:\s*[-_ ]?\s*(?:M|m|mo|month|months)?\s*0*([\d]+(?:\.[\d]+)?))?"
    r"\s*$",
    re.IGNORECASE
)

LR_C_GRID = [0.1, 1.0, 10.0]
RF_DEPTH_GRID = [10, 14, 18]
RF_LEAF_GRID = [3, 6]

LR_MAX_ITER = 400
RF_N_ESTIMATORS = 250


ALLOW_MULTIPLICITY = False

USE_GNN_PATIENT_UNIVERSE = True
GNN_FOLDS_BASE = DAMAGE_ROOT / "_final_gnn_folds"

OUTROOT = DAMAGE_ROOT / "baselines_lr_rf"
OUTROOT.mkdir(parents=True, exist_ok=True)

NUM_WHITELIST = [
    "length", "tortuosity",
    "radius_avg", "radius_min", "radius_max", "radius_SD", "vis_radius",
    "surface_area",
    "volume_mm3", "volume",
    "multiplicity",
    "dose_gy",
    "patient_tumor_volume_ml", "log1p_tumor_volume_ml", "patient_tumor_voxels",
]
BOOL_WHITELIST = ["is_ipsilateral", "is_in_tumor_lobe"]

LEAKAGE_SUBSTRINGS = ("final_label", "label", "damage", "damaged", "disappear", "surviv")


def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _parse_tp(tp: str):
    m = TP_RE.match(str(tp))
    if not m:
        return (np.nan, np.nan)
    fu_idx = float(m.group(1)) if m.group(1) else np.nan
    fu_mon = float(m.group(2)) if (m.group(2) not in (None, "")) else np.nan
    return fu_idx, fu_mon


def safe_to_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)


def _read_match_report(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["patient","modality","timepoint","match_frac"])
    rep = pd.read_csv(csv_path)
    cols = list(rep.columns)

    def _find(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    pcol = _find(["patient","pid","Patient"])
    mcol = _find(["modality","mod","Modality"])
    tcol = _find(["timepoint","tp","Timepoint","FU"])
    fcol = _find(["match_frac","match_fraction","matching_fraction","quality","match"])

    if pcol:
        rep = rep.rename(columns={pcol:"patient"})
    if mcol:
        rep = rep.rename(columns={mcol:"modality"})
    if tcol:
        rep = rep.rename(columns={tcol:"timepoint"})
    if fcol:
        rep = rep.rename(columns={fcol:"match_frac"})
    if "match_frac" not in rep.columns:
        rep["match_frac"] = np.nan

    rep["patient"] = rep.get("patient","").astype(str).str.strip()
    rep["modality"] = rep.get("modality","").astype(str).str.strip()
    rep["timepoint"] = rep.get("timepoint","").astype(str).str.strip()
    rep["match_frac"] = pd.to_numeric(rep["match_frac"], errors="coerce")
    return rep[["patient","modality","timepoint","match_frac"]]


def _collect_pairs_csv(thr_dir: Path, modality: str) -> pd.DataFrame:
    rows = []
    for pdir in sorted([d for d in thr_dir.glob("P*/") if d.is_dir()]):
        mdir = pdir / modality
        if not mdir.is_dir():
            continue
        for fu in sorted([d for d in mdir.iterdir() if d.is_dir() and d.name.upper().startswith("FU")]):
            rows_csv = fu / "edges_bl_fu_modeling_rows.csv"
            if rows_csv.exists():
                rows.append({
                    "patient": pdir.name,
                    "modality": modality,
                    "timepoint": fu.name,
                    "rows_csv": str(rows_csv),
                })
    return pd.DataFrame(rows)


def apply_qc(pairs: pd.DataFrame) -> pd.DataFrame:
    repdf = _read_match_report(REPORT_CSV)
    pairs = pairs.merge(repdf, how="left", on=["patient","modality","timepoint"])

    if ENABLE_QC and (not FORCE_KEEP_ALL_PAIRS):
        bad = pairs[pairs["match_frac"].notna() & (pairs["match_frac"] < MATCH_FRAC_THRESHOLD)].copy()
        bad_counts = bad.groupby("patient").size().to_dict()
        exclude_patients = sorted([p for p, c in bad_counts.items() if c >= BAD_GRAPHS_PATIENT_EXCLUDE])

        keep = pairs.copy()
        keep = keep[~keep["patient"].isin(exclude_patients)].copy()
        keep = keep[keep["match_frac"].isna() | (keep["match_frac"] >= MATCH_FRAC_THRESHOLD)].copy()

        log(f"[QC] removed_pairs={len(pairs)-len(keep)} | removed_patients={pairs['patient'].nunique()-keep['patient'].nunique()}")
        return keep

    log("[QC] skipped (keeping all pairs)")
    return pairs


def apply_hard_exclude(pairs: pd.DataFrame) -> pd.DataFrame:
    if not HARD_EXCLUDE_PATIENTS:
        return pairs
    before_p = pairs["patient"].nunique()
    before_n = len(pairs)
    keep = pairs[~pairs["patient"].isin(HARD_EXCLUDE_PATIENTS)].copy()
    log(f"[Hard exclude] removed_patients={before_p - keep['patient'].nunique()} | removed_pairs={before_n - len(keep)}")
    return keep


def make_patient_folds(patients_sorted, test_group_size=1, n_val=3, seed=SEED):
    pats = list(patients_sorted)
    if len(pats) < (n_val + test_group_size + 1):
        raise ValueError(f"Need at least {n_val + test_group_size + 1} patients, got {len(pats)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(pats)

    folds = []
    n_folds = int(math.ceil(len(pats) / float(test_group_size)))
    for i in range(n_folds):
        test = pats[i*test_group_size:(i+1)*test_group_size]
        if not test:
            continue
        remain = [p for p in pats if p not in test]
        rng_i = np.random.default_rng(seed + 1337 * (i + 1))
        rng_i.shuffle(remain)
        val = remain[:n_val]
        train = [p for p in remain if p not in val]
        folds.append({"rep": i, "train_pats": train, "val_pats": val, "test_pats": test})

    all_test = sum([r["test_pats"] for r in folds], [])
    assert len(all_test) == len(set(all_test)) == len(pats)
    return pd.DataFrame(folds)


def _stratified_subsample(idx, y, max_n, seed):
    if max_n is None or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    yb = y.astype(int)

    pos = idx[yb == 1]
    neg = idx[yb == 0]

    max_pos = int(round(max_n * float(SUBSAMPLE_KEEP_POS_FRACTION)))
    n_pos = min(len(pos), max_pos)
    n_neg = max_n - n_pos

    keep_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=np.int64)
    keep_neg = neg if len(neg) <= n_neg else rng.choice(neg, size=n_neg, replace=False)

    keep = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep)
    return np.sort(keep)


def micro_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def macro_metrics_per_patient(df_edges: pd.DataFrame):
    aucs, aps = [], []
    for _, g in df_edges.groupby("patient"):
        y = g["y"].astype(int).values
        p = g["p"].astype(float).values
        if np.unique(y).size < 2:
            continue
        aucs.append(roc_auc_score(y, p))
        aps.append(average_precision_score(y, p))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.mean(aps))


def make_lr(C, rep_seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=LR_MAX_ITER,
            C=float(C),
            class_weight=LR_CLASS_WEIGHT,
            random_state=rep_seed,
        )),
    ])


def make_rf(max_depth, min_leaf, rep_seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=int(max_depth),
            min_samples_leaf=int(min_leaf),
            n_jobs=-1,
            class_weight=RF_CLASS_WEIGHT,
            random_state=rep_seed,
        )),
    ])


def load_gnn_patient_universe(modality: str, thr: str):
    gnn_folds_path = GNN_FOLDS_BASE / thr / modality / "folds.csv"
    if not gnn_folds_path.exists():
        return None, gnn_folds_path

    df = pd.read_csv(gnn_folds_path)
    if "test_pats" not in df.columns:
        return None, gnn_folds_path

    pats = set()
    for s in df["test_pats"].dropna().astype(str).values:
        for p in s.split(","):
            p = p.strip()
            if p:
                pats.add(p)

    if not pats:
        return None, gnn_folds_path

    return sorted(pats), gnn_folds_path


def build_edge_table_from_csvs(keep_pairs: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rows_out = []
    pat_edge_counts = defaultdict(int)

    num_whitelist = list(NUM_WHITELIST)
    if not ALLOW_MULTIPLICITY and "multiplicity" in num_whitelist:
        num_whitelist.remove("multiplicity")

    for r in keep_pairs.itertuples(index=False):
        try:
            df = pd.read_csv(r.rows_csv)
        except Exception:
            continue

        if "label_damaged" not in df.columns:
            continue

        df["patient"] = str(r.patient)
        df["timepoint"] = str(r.timepoint)

        if ADD_FU_TIME:
            fu_idx, fu_mon = _parse_tp(r.timepoint)
            df["fu_index"] = fu_idx
            df["fu_months"] = fu_mon

        y = pd.to_numeric(df["label_damaged"], errors="coerce").astype("Int64")
        m = y.isin([0, 1]).to_numpy()
        if m.sum() < 5:
            continue
        df = df.loc[m].copy()
        df["y"] = y.loc[m].astype(int).to_numpy()

        drop_cols = [c for c in df.columns if any(s in c.lower() for s in LEAKAGE_SUBSTRINGS)]
        drop_cols = [c for c in drop_cols if c != "y"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        present_num = [c for c in num_whitelist if c in df.columns]
        present_bool = [c for c in BOOL_WHITELIST if c in df.columns]

        if "volume_mm3" not in present_num and "volume" in df.columns:
            df["volume_mm3"] = pd.to_numeric(df["volume"], errors="coerce")
            present_num = [c for c in present_num if c != "volume"]
            present_num.append("volume_mm3")

        if ADD_FU_TIME:
            for c in ["fu_index", "fu_months"]:
                if c in df.columns and c not in present_num:
                    present_num.append(c)

        if not present_num and not present_bool:
            continue

        for c in present_num + present_bool:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        idx = np.arange(len(df), dtype=np.int64)
        if MAX_EDGES_PER_PAIR is not None and len(idx) > MAX_EDGES_PER_PAIR:
            y_np = df["y"].to_numpy(dtype=int)
            if SUBSAMPLE_STRATIFIED:
                idx = _stratified_subsample(idx, y_np, MAX_EDGES_PER_PAIR, seed=SEED)
            else:
                rng = np.random.default_rng(SEED)
                idx = rng.choice(idx, size=MAX_EDGES_PER_PAIR, replace=False)
                idx.sort()
            df = df.iloc[idx].copy()

        if MAX_EDGES_PER_PATIENT is not None:
            nE = len(df)
            if pat_edge_counts[r.patient] + nE > MAX_EDGES_PER_PATIENT:
                continue
            pat_edge_counts[r.patient] += nE

        keep_cols = ["patient", "y"] + present_num + present_bool
        rows_out.append(df[keep_cols].copy())

    if not rows_out:
        return pd.DataFrame()

    out = pd.concat(rows_out, ignore_index=True)
    log(f"[Cache CSV] edges={len(out)} | patients={out['patient'].nunique()} | time={(time.time()-t0):.1f}s")
    return out


def run_one(modality: str, thr: str):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    thr_dir = DAMAGE_ROOT / thr
    pairs = _collect_pairs_csv(thr_dir, modality)
    log(f"\n================ THR={thr} | MOD={modality} ================")
    log(f"Pairs found (pre-QC): {len(pairs)} | Patients: {pairs['patient'].nunique()}")
    if pairs.empty:
        return

    keep_pairs = apply_qc(pairs)
    keep_pairs = apply_hard_exclude(keep_pairs)
    log(f"Pairs kept (post-QC+exclude): {len(keep_pairs)} | Patients kept: {keep_pairs['patient'].nunique()}")
    if keep_pairs["patient"].nunique() < (N_VAL_PATIENTS + TEST_GROUP_SIZE + 1):
        log("Not enough patients after QC.")
        return

    df_all = build_edge_table_from_csvs(keep_pairs)
    if df_all.empty:
        log("No usable edges from CSVs.")
        return

    if USE_GNN_PATIENT_UNIVERSE:
        gnn_pats, gnn_path = load_gnn_patient_universe(modality, thr)
        if gnn_pats is None:
            log(f"[Align] GNN folds not found or unusable at: {gnn_path}")
            patients_sorted = sorted(df_all["patient"].unique())
        else:
            base_pats = sorted(df_all["patient"].unique())
            base_set = set(base_pats)
            gnn_set = set(gnn_pats)

            missing_in_baseline = sorted(list(gnn_set - base_set))
            extra_in_baseline = sorted(list(base_set - gnn_set))

            if missing_in_baseline:
                log(f"[Align] WARNING: {len(missing_in_baseline)} GNN patients missing in CSV baseline (no edges after filters). "
                    f"Will drop them from fold universe. Example: {missing_in_baseline[:8]}")

            if extra_in_baseline:
                log(f"[Align] Dropping {len(extra_in_baseline)} baseline-only patients to match GNN universe. Example: {extra_in_baseline[:8]}")

            keep_patients = sorted(list(base_set & gnn_set))
            df_all = df_all[df_all["patient"].isin(keep_patients)].copy()
            patients_sorted = sorted(keep_patients)
            log(f"[Align] Using GNN-aligned patient universe: {len(patients_sorted)} patients")
    else:
        patients_sorted = sorted(df_all["patient"].unique())

    if len(patients_sorted) < (N_VAL_PATIENTS + TEST_GROUP_SIZE + 1):
        log("Not enough patients after alignment.")
        return

    folds = make_patient_folds(patients_sorted, test_group_size=TEST_GROUP_SIZE, n_val=N_VAL_PATIENTS, seed=SEED)

    outdir = OUTROOT / modality / thr
    outdir.mkdir(parents=True, exist_ok=True)

    folds_out = folds.copy()
    folds_out["train_pats"] = folds_out["train_pats"].apply(lambda x: ",".join(x))
    folds_out["val_pats"]   = folds_out["val_pats"].apply(lambda x: ",".join(x))
    folds_out["test_pats"]  = folds_out["test_pats"].apply(lambda x: ",".join(x))
    safe_to_csv(folds_out, outdir / "cv_folds_patientwise.csv")

    feature_cols = [c for c in df_all.columns if c not in ("patient", "y")]
    bad_cols = [c for c in feature_cols if any(s in c.lower() for s in LEAKAGE_SUBSTRINGS)]
    if bad_cols:
        raise RuntimeError(f"Leakage-like columns survived filtering: {bad_cols}")

    rows = []

    for fold in folds.itertuples(index=False):
        rep = int(fold.rep)
        rep_seed = SEED + rep * 1337

        train_p = set(fold.train_pats)
        val_p   = set(fold.val_pats)
        test_p  = set(fold.test_pats)

        df_tr = df_all[df_all["patient"].isin(train_p)].copy()
        df_va = df_all[df_all["patient"].isin(val_p)].copy()
        df_te = df_all[df_all["patient"].isin(test_p)].copy()
        if df_tr.empty or df_va.empty or df_te.empty:
            continue

        X_tr = df_tr[feature_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr["y"].to_numpy(dtype=int)

        X_va = df_va[feature_cols].to_numpy(dtype=np.float32)
        y_va = df_va["y"].to_numpy(dtype=int)

        X_te = df_te[feature_cols].to_numpy(dtype=np.float32)
        y_te = df_te["y"].to_numpy(dtype=int)

        best_lr = None
        best_lr_val = -1e18
        best_lr_C = None

        for C in LR_C_GRID:
            lr = make_lr(C, rep_seed)
            lr.fit(X_tr, y_tr)
            p_va = lr.predict_proba(X_va)[:, 1]
            df_va_edges = pd.DataFrame({"patient": df_va["patient"].values, "y": y_va, "p": p_va})
            va_macro_auc, _ = macro_metrics_per_patient(df_va_edges)
            score = va_macro_auc if np.isfinite(va_macro_auc) else -1e18
            if score > best_lr_val:
                best_lr_val = score
                best_lr = lr
                best_lr_C = C

        p_te = best_lr.predict_proba(X_te)[:, 1]
        te_micro_auc, te_micro_ap = micro_metrics(y_te, p_te)
        df_te_edges = pd.DataFrame({"patient": df_te["patient"].values, "y": y_te, "p": p_te})
        te_macro_auc, te_macro_ap = macro_metrics_per_patient(df_te_edges)

        rows.append({
            "thr": thr, "modality": modality, "rep": rep, "model": "LogReg",
            "chosen_params": f"C={best_lr_C}",
            "val_macro_auc": float(best_lr_val),
            "test_macro_auc": float(te_macro_auc),
            "test_macro_ap": float(te_macro_ap),
            "test_micro_auc": float(te_micro_auc),
            "test_micro_ap": float(te_micro_ap),
            "n_train_edges": int(len(df_tr)),
            "n_val_edges": int(len(df_va)),
            "n_test_edges": int(len(df_te)),
            "test_pats": ",".join(sorted(test_p)),
        })

        best_rf = None
        best_rf_val = -1e18
        best_rf_cfg = None

        for d in RF_DEPTH_GRID:
            for leaf in RF_LEAF_GRID:
                rf = make_rf(d, leaf, rep_seed)
                rf.fit(X_tr, y_tr)
                p_va = rf.predict_proba(X_va)[:, 1]
                df_va_edges = pd.DataFrame({"patient": df_va["patient"].values, "y": y_va, "p": p_va})
                va_macro_auc, _ = macro_metrics_per_patient(df_va_edges)
                score = va_macro_auc if np.isfinite(va_macro_auc) else -1e18
                if score > best_rf_val:
                    best_rf_val = score
                    best_rf = rf
                    best_rf_cfg = (d, leaf)

        p_te = best_rf.predict_proba(X_te)[:, 1]
        te_micro_auc, te_micro_ap = micro_metrics(y_te, p_te)
        df_te_edges = pd.DataFrame({"patient": df_te["patient"].values, "y": y_te, "p": p_te})
        te_macro_auc, te_macro_ap = macro_metrics_per_patient(df_te_edges)

        rows.append({
            "thr": thr, "modality": modality, "rep": rep, "model": "RandomForest",
            "chosen_params": f"max_depth={best_rf_cfg[0]},min_leaf={best_rf_cfg[1]}",
            "val_macro_auc": float(best_rf_val),
            "test_macro_auc": float(te_macro_auc),
            "test_macro_ap": float(te_macro_ap),
            "test_micro_auc": float(te_micro_auc),
            "test_micro_ap": float(te_micro_ap),
            "n_train_edges": int(len(df_tr)),
            "n_val_edges": int(len(df_va)),
            "n_test_edges": int(len(df_te)),
            "test_pats": ",".join(sorted(test_p)),
        })

        log(f"[{thr}|{modality}] rep={rep:02d} test={','.join(sorted(test_p))} "
            f"LR(val macroAUC={best_lr_val:.3f}, C={best_lr_C}) | "
            f"RF(val macroAUC={best_rf_val:.3f}, cfg={best_rf_cfg})")

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        log("No results produced.")
        return

    safe_to_csv(df_res, outdir / "folds.csv")

    summ = (df_res.groupby("model")
            .agg(test_macro_auc_mean=("test_macro_auc","mean"),
                 test_macro_auc_std=("test_macro_auc","std"),
                 test_macro_ap_mean=("test_macro_ap","mean"),
                 test_macro_ap_std=("test_macro_ap","std"),
                 test_micro_auc_mean=("test_micro_auc","mean"),
                 test_micro_ap_mean=("test_micro_ap","mean"),
                 reps=("rep","count"))
            .reset_index()
            .sort_values("test_macro_auc_mean", ascending=False))
    safe_to_csv(summ, outdir / "summary.csv")

    log(f"[{thr}|{modality}] Summary saved → {outdir / 'summary.csv'}")
    for r in summ.itertuples(index=False):
        log(f"  {r.model:12s} | TEST macroAUC={r.test_macro_auc_mean:.3f}±{r.test_macro_auc_std:.3f} "
            f"| macroAP={r.test_macro_ap_mean:.3f}±{r.test_macro_ap_std:.3f} "
            f"| microAUC={r.test_micro_auc_mean:.3f} microAP={r.test_micro_ap_mean:.3f} | reps={r.reps}")


def main():
    set_seed(SEED)
    log(f"MODALITIES={MODALITIES} | THRESHOLDS={THRESHOLDS}")
    log(f"QC: ENABLE_QC={ENABLE_QC} FORCE_KEEP_ALL_PAIRS={FORCE_KEEP_ALL_PAIRS} "
        f"| MATCH_FRAC_THRESHOLD={MATCH_FRAC_THRESHOLD} BAD_GRAPHS_PATIENT_EXCLUDE={BAD_GRAPHS_PATIENT_EXCLUDE}")
    log(f"Hard exclude: {len(HARD_EXCLUDE_PATIENTS)} patients")
    log(f"Folds: LOPO test_group_size={TEST_GROUP_SIZE} | N_VAL_PATIENTS={N_VAL_PATIENTS}")
    log(f"Caps: MAX_EDGES_PER_PAIR={MAX_EDGES_PER_PAIR} | MAX_EDGES_PER_PATIENT={MAX_EDGES_PER_PATIENT}")
    log(f"Sampling: STRATIFIED={SUBSAMPLE_STRATIFIED} keep_pos_frac={SUBSAMPLE_KEEP_POS_FRACTION}")
    log(f"Whitelist: NUM={len(NUM_WHITELIST)} BOOL={len(BOOL_WHITELIST)} | ALLOW_MULTIPLICITY={ALLOW_MULTIPLICITY}")
    log(f"Time as feature: {ADD_FU_TIME}")
    log(f"VAL selection grids: LR_C={LR_C_GRID} | RF_depth={RF_DEPTH_GRID} RF_leaf={RF_LEAF_GRID}")
    log(f"Align to GNN folds: {USE_GNN_PATIENT_UNIVERSE} | GNN_FOLDS_BASE={GNN_FOLDS_BASE}")

    for mod in MODALITIES:
        for thr in THRESHOLDS:
            t0 = time.time()
            run_one(mod, thr)
            log(f"[{thr}|{mod}] done in {(time.time()-t0):.1f}s")

    log("\nDone.")


if __name__ == "__main__":
    main()
