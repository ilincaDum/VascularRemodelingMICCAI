"""
Line-graph residual-gated GNN 

Runs both modalities (Artery, Vein) over multiple damage thresholds using
LOPO-style patient folds. Uses full line-graph, with an
optional cap on incoming line-graph degree to control message passing cost

Inputs:
- Labeled GraphML pairs produced by the matching/labeling pipeline:
  <ROOT>/final_labeled_graphs/damage_thresholds/<thr>/P*/<Mod>/FU*/BL_labeled.graphml

Analyses (per threshold, per modality):
A) Ground-truth trend: damage rate vs dose bins across all edges 
B) Post-hoc calibration: on aggregated TEST predictions, bin by dose and compare
   mean predicted probability vs empirical injury rate

Outputs per (thr, modality):
<ROOT>/final_labeled_graphs/damage_thresholds/_full_no_film/<thr>/<mod>/
    folds.csv
    summary.csv
    dose_damage_trend.csv
    dose_calibration_full_no_film.csv
"""

import os, re, math, random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(r"C:\Users\ilinc\OneDrive\Desktop\GraphAnalysis\GraphsCompleteAnalysis\graphs_complete_cleaned")
FLG = ROOT / "final_labeled_graphs"
DAMAGE_ROOT = FLG / "damage_thresholds"

THRESHOLDS = ["damage_90pct", "damage_75pct", "damage_50pct"]
MODALITIES = ["Artery", "Vein"]

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAP_LG_IN_DEG = 24
PRECACHE_TENSORS_PER_FOLD = True
USE_AMP = True

EPOCHS = 20
PATIENCE = 6
LR_NEURAL = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0
DROPOUT = 0.15
HID = 128
ROUNDS = 2
FOCAL_GAMMA = 1.0
EARLYSTOP_W_AUC = 0.5
EARLYSTOP_W_AP = 0.5

TEST_GROUP_SIZE = 1
N_VAL_PATIENTS = 3
LIMIT_FOLDS = None

ENABLE_QC = True
MATCH_FRAC_THRESHOLD = 0.75
BAD_GRAPHS_PATIENT_EXCLUDE = 2
REPORT_CSV = FLG / "_postmatch_reports" / "lobe_counts_and_quality.csv"

HARD_EXCLUDE_PATIENTS = {
    "P144", "P151", "P152", "P155", "P168", "P172", "P181", "P183", "P190", "P109"
}

MAX_EDGES_PER_PAIR = 15000
MAX_EDGES_PER_PATIENT = 120000
STRATIFIED_PAIR_SUBSAMPLE = True
KEEP_POS_FRACTION_WHEN_CAP = 0.50

DOSE_BIN_WIDTH_GY = 5.0
MIN_BINS = 6

OUTBASE = DAMAGE_ROOT / "_full_no_film"
OUTBASE.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    import builtins
    with builtins.open(path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)


def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LEAK_SUBSTR = ("label", "final_label", "damage", "damaged", "disappear", "surviv")


def _is_leaky_key(k: str) -> bool:
    s = str(k).strip().lower()
    return any(t in s for t in LEAK_SUBSTR)


TP_RE = re.compile(
    r"FU\s*0*(\d+)"
    r"(?:\s*[-_ ]?\s*(?:M|m|mo|month|months)?\s*0*([\d]+(?:\.[\d]+)?))?"
    r"\s*$",
    re.IGNORECASE
)


def _parse_tp(tp: str):
    m = TP_RE.match(str(tp))
    if not m:
        return (np.nan, np.nan)
    fu_idx = float(m.group(1)) if m.group(1) else np.nan
    fu_mon = float(m.group(2)) if (m.group(2) not in (None, "")) else np.nan
    return fu_idx, fu_mon


EDGE_NUM_KEYS_BASE = [
    "length", "tortuosity", "radius_avg", "radius_min", "radius_max", "radius_SD", "vis_radius",
    "surface_area", "volume", "dose_gy",
    "patient_tumor_volume_ml", "log1p_tumor_volume_ml", "patient_tumor_voxels",
]
EDGE_BOOL_KEYS_BASE = ["is_ipsilateral", "is_in_tumor_lobe"]
EDGE_CAT_KEYS_BASE = ["lobe", "side"]


def _as_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _as_bool01(x):
    if isinstance(x, (bool, np.bool_)):
        return 1.0 if x else 0.0
    s = str(x).strip().lower()
    if s in ("1", "true", "yes"):
        return 1.0
    if s in ("0", "false", "no"):
        return 0.0
    return np.nan


def _parse_final_label_to_y(final_label):
    s = "" if final_label is None else str(final_label).strip().lower()
    if s in ("damaged", "disappeared"):
        return 1.0
    if s in ("survived",):
        return 0.0
    return np.nan


def _metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def macro_auc_ap_per_patient(pat_ids, y, p):
    df = pd.DataFrame({"patient": np.asarray(pat_ids, object),
                       "y": np.asarray(y, int),
                       "p": np.asarray(p, float)})
    aucs, aps = [], []
    for _, g in df.groupby("patient"):
        yy, pp = g["y"].values, g["p"].values
        if np.unique(yy).size < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.mean(aps))


def eval_from_arrays(PAT, Y, P):
    micro_auc, micro_ap = _metrics(Y, P)
    macro_auc, macro_ap = macro_auc_ap_per_patient(PAT, Y, P)
    pos_rate = float(np.mean(Y)) if len(Y) else float("nan")
    return macro_auc, macro_ap, micro_auc, micro_ap, pos_rate


def _read_match_report(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["patient", "modality", "timepoint", "match_frac"])
    rep = pd.read_csv(csv_path)
    cols = list(rep.columns)

    def _find(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    pcol = _find(["patient", "pid", "Patient"])
    mcol = _find(["modality", "mod", "Modality"])
    tcol = _find(["timepoint", "tp", "Timepoint", "FU"])
    fcol = _find(["match_frac", "match_fraction", "matching_fraction", "quality", "match"])

    if pcol:
        rep = rep.rename(columns={pcol: "patient"})
    if mcol:
        rep = rep.rename(columns={mcol: "modality"})
    if tcol:
        rep = rep.rename(columns={tcol: "timepoint"})
    if fcol:
        rep = rep.rename(columns={fcol: "match_frac"})
    if "match_frac" not in rep.columns:
        rep["match_frac"] = np.nan

    rep["patient"] = rep.get("patient", "").astype(str).str.strip()
    rep["modality"] = rep.get("modality", "").astype(str).str.strip()
    rep["timepoint"] = rep.get("timepoint", "").astype(str).str.strip()
    rep["match_frac"] = pd.to_numeric(rep["match_frac"], errors="coerce")
    return rep[["patient", "modality", "timepoint", "match_frac"]]


def _collect_pairs_graphml(thr_dir: Path, modality: str) -> pd.DataFrame:
    rows = []
    for pdir in sorted([d for d in thr_dir.glob("P*/") if d.is_dir()]):
        pat = pdir.name
        mdir = pdir / modality
        if not mdir.is_dir():
            continue
        for fu in sorted([d for d in mdir.iterdir() if d.is_dir() and d.name.upper().startswith("FU")]):
            gpath = fu / "BL_labeled.graphml"
            if gpath.exists():
                rows.append({"patient": pat, "modality": modality, "timepoint": fu.name, "graphml": str(gpath)})
    return pd.DataFrame(rows)


def apply_qc(pairs: pd.DataFrame) -> pd.DataFrame:
    repdf = _read_match_report(REPORT_CSV)
    pairs = pairs.merge(repdf, how="left", on=["patient", "modality", "timepoint"])
    if not ENABLE_QC:
        return pairs

    bad = pairs[pairs["match_frac"].notna() & (pairs["match_frac"] < MATCH_FRAC_THRESHOLD)].copy()
    bad_counts = bad.groupby("patient").size().to_dict()
    exclude_patients = sorted([p for p, c in bad_counts.items() if c >= BAD_GRAPHS_PATIENT_EXCLUDE])

    keep = pairs.copy()
    keep = keep[~keep["patient"].isin(exclude_patients)].copy()
    keep = keep[keep["match_frac"].isna() | (keep["match_frac"] >= MATCH_FRAC_THRESHOLD)].copy()

    log(f"[QC] removed_pairs={len(pairs)-len(keep)} | removed_patients={pairs['patient'].nunique()-keep['patient'].nunique()}")
    return keep


def make_patient_folds(patients_sorted, test_group_size=1, n_val=3, seed=SEED):
    pats = list(patients_sorted)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)

    folds = []
    n_folds = int(math.ceil(len(pats) / float(test_group_size)))
    for i in range(n_folds):
        test = pats[i * test_group_size:(i + 1) * test_group_size]
        if not test:
            continue
        remain = [p for p in pats if p not in test]
        rng_i = np.random.default_rng(seed + 1337 * (i + 1))
        rng_i.shuffle(remain)
        val = remain[:n_val]
        train = [p for p in remain if p not in val]
        folds.append({"rep": i, "train": train, "val": val, "test": test})

    all_test = sum([r["test"] for r in folds], [])
    assert len(all_test) == len(set(all_test)) == len(pats)
    return folds


def fit_norm_cont_from_fullX(cache, pair_keys):
    sumv = None
    sumsq = None
    cnt = None
    for pk in pair_keys:
        X = cache[pk]["X_cont_full"].astype(np.float32)
        m = np.isfinite(X)
        X0 = np.where(m, X, 0.0)
        if sumv is None:
            d = X.shape[1]
            sumv = np.zeros((d,), np.float64)
            sumsq = np.zeros((d,), np.float64)
            cnt = np.zeros((d,), np.float64)
        sumv += X0.sum(axis=0)
        sumsq += (X0 * X0).sum(axis=0)
        cnt += m.sum(axis=0)

    cnt = np.maximum(cnt, 1.0)
    mu = (sumv / cnt).astype(np.float32)
    var = (sumsq / cnt) - (mu.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-6)
    sd = np.sqrt(var).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    return mu, sd


def fit_norm_1d(values_list):
    x = np.concatenate(values_list, axis=0) if values_list else np.zeros((0,), np.float32)
    mu = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    sd = float(np.nanstd(x)) if np.isfinite(x).any() else 1.0
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    return np.float32(mu), np.float32(sd)


def build_vocab_from_train(cache, pair_keys, field):
    vocab = {"": 0}
    for pk in pair_keys:
        arr = cache[pk]["edge_cat_raw"][field]
        for v in arr:
            s = "" if v is None else str(v)
            if s not in vocab:
                vocab[s] = len(vocab)
    return vocab


def _stratified_cap_indices(y, max_n, seed):
    y = y.astype(int)
    idx = np.arange(len(y), dtype=np.int64)
    if len(idx) <= max_n:
        return idx

    rng = np.random.default_rng(seed)
    pos = idx[y == 1]
    neg = idx[y == 0]

    max_pos = int(round(max_n * float(KEEP_POS_FRACTION_WHEN_CAP)))
    n_pos = min(len(pos), max_pos)
    n_neg = max_n - n_pos

    keep_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=np.int64)
    keep_neg = neg if len(neg) <= n_neg else rng.choice(neg, size=n_neg, replace=False)

    keep = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep)
    keep.sort()
    return keep


def build_linegraph_edge_index_none(u, v, n_nodes):
    inc = [[] for _ in range(int(n_nodes))]
    for ei, (uu, vv) in enumerate(zip(u, v)):
        inc[int(uu)].append(ei)
        inc[int(vv)].append(ei)

    src, dst = [], []
    for lst in inc:
        L = len(lst)
        if L < 2:
            continue
        for i in range(L):
            a = lst[i]
            for j in range(L):
                if i == j:
                    continue
                src.append(a)
                dst.append(lst[j])

    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.asarray(src, np.int64), np.asarray(dst, np.int64)], axis=0)


def cap_edge_index_per_dst(edge_index, E, K, seed=42):
    if edge_index.shape[1] == 0 or K is None:
        return edge_index
    rng = np.random.default_rng(seed)
    src = edge_index[0]
    dst = edge_index[1]

    buckets = [[] for _ in range(E)]
    for j in range(dst.shape[0]):
        buckets[int(dst[j])].append(j)

    keep_cols = []
    for d in range(E):
        cols = buckets[d]
        if len(cols) <= K:
            keep_cols.extend(cols)
        else:
            sel = rng.choice(np.asarray(cols, np.int64), size=K, replace=False)
            keep_cols.extend(sel.tolist())

    keep_cols = np.asarray(keep_cols, np.int64)
    keep_cols.sort()
    return np.stack([src[keep_cols], dst[keep_cols]], axis=0)


def graphml_to_pair_cache(graphml_path: str, patient: str, timepoint: str,
                          edge_num_keys_full, edge_bool_keys, edge_cat_keys,
                          seed: int):
    G = nx.read_graphml(graphml_path)
    fu_idx, _ = _parse_tp(timepoint)

    node_ids = list(G.nodes())
    nid2i = {str(n): i for i, n in enumerate(node_ids)}

    u_idx, v_idx = [], []
    X_cont, X_bool = [], []
    e_cat_raw = {k: [] for k in edge_cat_keys}
    y_list = []

    dose_raw = []
    fuidx_raw = []

    for u, v, ed in G.edges(data=True):
        y = _parse_final_label_to_y(ed.get("final_label", None))
        if not np.isfinite(y):
            continue

        vals = [_as_float(ed.get(k, np.nan)) for k in edge_num_keys_full]
        bvals = [_as_bool01(ed.get(k, np.nan)) for k in edge_bool_keys]

        X_cont.append(vals)
        X_bool.append(bvals)
        y_list.append(float(y))

        for k in edge_cat_keys:
            e_cat_raw[k].append("" if ed.get(k, None) is None else str(ed.get(k)))

        u_idx.append(nid2i[str(u)])
        v_idx.append(nid2i[str(v)])

        d = _as_float(ed.get("dose_gy", np.nan))
        dose_raw.append(d)
        fuidx_raw.append(float(fu_idx) if np.isfinite(fu_idx) else np.nan)

    if len(y_list) < 5:
        return None

    X_cont = np.asarray(X_cont, np.float32)
    X_bool = np.asarray(X_bool, np.float32)
    y = np.asarray(y_list, np.float32)
    u_idx = np.asarray(u_idx, np.int64)
    v_idx = np.asarray(v_idx, np.int64)
    n_nodes = len(node_ids)

    dose_raw = np.asarray(dose_raw, np.float32)
    fuidx_raw = np.asarray(fuidx_raw, np.float32)

    idx = np.arange(len(y), dtype=np.int64)
    if MAX_EDGES_PER_PAIR is not None and idx.size > int(MAX_EDGES_PER_PAIR):
        if STRATIFIED_PAIR_SUBSAMPLE:
            idx = _stratified_cap_indices(y, int(MAX_EDGES_PER_PAIR), seed=seed)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=int(MAX_EDGES_PER_PAIR), replace=False)
            idx.sort()

        X_cont = X_cont[idx]
        X_bool = X_bool[idx]
        y = y[idx]
        u_idx = u_idx[idx]
        v_idx = v_idx[idx]
        dose_raw = dose_raw[idx]
        fuidx_raw = fuidx_raw[idx]
        for k in edge_cat_keys:
            e_cat_raw[k] = [e_cat_raw[k][i] for i in idx.tolist()]

    edge_index_none = build_linegraph_edge_index_none(u_idx, v_idx, n_nodes)
    if CAP_LG_IN_DEG is not None:
        edge_index_none = cap_edge_index_per_dst(edge_index_none, E=len(y), K=int(CAP_LG_IN_DEG), seed=seed)

    return dict(
        patient=patient,
        pair_key=f"{patient}|{timepoint}",
        timepoint=timepoint,
        X_cont_full=X_cont,
        X_bool=X_bool,
        y=y,
        n_nodes=int(n_nodes),
        edge_cat_raw={k: np.asarray(vv, object) for k, vv in e_cat_raw.items()},
        dose_raw=dose_raw,
        fuidx_raw=fuidx_raw,
        edge_index_none=edge_index_none
    )


def build_pair_cache(pairs: pd.DataFrame, edge_num_keys_full, edge_bool_keys, edge_cat_keys):
    cache = {}
    by_pat = defaultdict(list)
    pat_edge_counts = defaultdict(int)
    skipped = 0

    for r in pairs.itertuples(index=False):
        rec = graphml_to_pair_cache(
            r.graphml, r.patient, r.timepoint,
            edge_num_keys_full=edge_num_keys_full,
            edge_bool_keys=edge_bool_keys,
            edge_cat_keys=edge_cat_keys,
            seed=SEED
        )
        if rec is None:
            skipped += 1
            continue

        nE = len(rec["y"])
        if MAX_EDGES_PER_PATIENT is not None:
            if pat_edge_counts[r.patient] + nE > int(MAX_EDGES_PER_PATIENT):
                continue
            pat_edge_counts[r.patient] += nE

        cache[rec["pair_key"]] = rec
        by_pat[r.patient].append(rec["pair_key"])

    log(f"[Cache] patients={len(by_pat)} | pairs={len(cache)} | skipped_pairs={skipped}")
    return cache, by_pat


def _dose_bins_from_data(dose_values, width=5.0, min_bins=6):
    dmax = float(np.nanmax(dose_values)) if np.isfinite(dose_values).any() else 0.0
    if dmax <= 0:
        return np.array([0.0, 1.0], dtype=np.float32)
    hi = math.ceil(dmax / width) * width
    nb = int(max(min_bins, hi / width))
    edges = np.arange(0.0, (nb * width) + width, width, dtype=np.float32)
    if edges.size < 2:
        edges = np.array([0.0, max(1.0, dmax)], dtype=np.float32)
    return edges


def dose_damage_trend_from_cache(cache, out_csv: Path, bin_edges):
    ys = []
    ds = []
    for _, rec in cache.items():
        y = rec["y"].astype(np.int32)
        d = rec["dose_raw"].astype(np.float32)
        m = np.isfinite(d)
        ys.append(y[m])
        ds.append(d[m])
    if not ys:
        return None

    Y = np.concatenate(ys)
    D = np.concatenate(ds)
    if D.size < 10:
        return None

    df = pd.DataFrame({"dose_gy": D, "y": Y})
    df["dose_bin"] = pd.cut(df["dose_gy"], bins=bin_edges, include_lowest=True)
    df["dose_bin"] = df["dose_bin"].astype(str)

    g = (df.groupby("dose_bin", observed=True)
           .agg(n=("y", "size"),
                damage_rate=("y", "mean"),
                dose_median=("dose_gy", "median"),
                dose_mean=("dose_gy", "mean"))
           .reset_index()
           .sort_values("dose_median"))

    safe_to_csv(g, out_csv)

    if len(g) >= 3:
        corr = float(np.corrcoef(g["dose_median"].values, g["damage_rate"].values)[0, 1])
    else:
        corr = float("nan")
    return g, corr


def dose_calibration_from_predictions(PAT, Y, P, DOSE, out_csv: Path, bin_edges):
    df = pd.DataFrame({
        "patient": np.asarray(PAT, object),
        "y": np.asarray(Y, int),
        "p": np.asarray(P, float),
        "dose_gy": np.asarray(DOSE, float),
    })
    df = df[np.isfinite(df["dose_gy"].values)].copy()
    if len(df) < 50:
        return None

    df["dose_bin"] = pd.cut(df["dose_gy"], bins=bin_edges, include_lowest=True)
    df["dose_bin"] = df["dose_bin"].astype(str)

    g = (df.groupby("dose_bin", observed=True)
           .agg(n=("y", "size"),
                emp_damage_rate=("y", "mean"),
                mean_pred=("p", "mean"),
                dose_median=("dose_gy", "median"),
                dose_mean=("dose_gy", "mean"))
           .reset_index()
           .sort_values("dose_median"))

    safe_to_csv(g, out_csv)
    return g


def materialize_pair_torch(rec,
                           mu_c_full, sd_c_full,
                           lobe_vocab, side_vocab,
                           time_mu, time_sd):
    Xc = (rec["X_cont_full"].astype(np.float32) - mu_c_full) / sd_c_full
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)

    Xb = np.nan_to_num(rec["X_bool"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    parts = [Xc, Xb]

    t = rec["fuidx_raw"].astype(np.float32)
    tt = (t - float(time_mu)) / float(time_sd)
    tok = np.isfinite(t).astype(np.float32)
    tt = np.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)
    tt = (tt * tok).astype(np.float32)
    parts.append(tt[:, None])

    lobe_raw = rec["edge_cat_raw"]["lobe"]
    side_raw = rec["edge_cat_raw"]["side"]
    l_ids = np.asarray([lobe_vocab.get("" if v is None else str(v), 0) for v in lobe_raw], np.int64)
    s_ids = np.asarray([side_vocab.get("" if v is None else str(v), 0) for v in side_raw], np.int64)
    l_oh = np.zeros((len(l_ids), len(lobe_vocab)), np.float32)
    l_oh[np.arange(len(l_ids)), l_ids] = 1.0
    s_oh = np.zeros((len(s_ids), len(side_vocab)), np.float32)
    s_oh[np.arange(len(s_ids)), s_ids] = 1.0
    parts += [l_oh, s_oh]

    X = np.concatenate(parts, axis=1).astype(np.float32)
    y = rec["y"].astype(np.float32)

    edge_index_np = rec["edge_index_none"]
    mp_mask = np.ones((len(y),), np.float32)

    Xt = torch.from_numpy(X).float().to(DEVICE)
    yt = torch.from_numpy(y).float().to(DEVICE)
    edge_index = torch.from_numpy(edge_index_np).long().to(DEVICE)
    mp_mask_t = torch.from_numpy(mp_mask).float().to(DEVICE)

    return Xt, yt, edge_index, mp_mask_t


class WeightedFocalBCE(nn.Module):
    def __init__(self, gamma=0.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits, y, pos_weight):
        if self.gamma <= 0:
            w = torch.where(y > 0.5, pos_weight, torch.ones_like(y))
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            return (w * bce).mean()

        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        pt = p * y + (1 - p) * (1 - y)
        focal = (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        w = torch.where(y > 0.5, pos_weight, torch.ones_like(y))
        return (w * focal * bce).mean()


def compute_pos_weight_from_pairs(cache, pair_keys):
    ys = [cache[pk]["y"].astype(np.float32) for pk in pair_keys]
    y = np.concatenate(ys).astype(int) if ys else np.array([], int)
    if y.size == 0:
        return 1.0
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(np.clip(n_neg / max(1, n_pos), 1.0, 12.0))


def scatter_mean_edge(h_src, edge_index, E):
    if edge_index.numel() == 0:
        return torch.zeros((E, h_src.size(1)), device=h_src.device, dtype=h_src.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    H = h_src.size(1)
    agg = torch.zeros((E, H), device=h_src.device, dtype=h_src.dtype)
    deg = torch.zeros((E,), device=h_src.device, dtype=h_src.dtype)
    agg.index_add_(0, dst, h_src[src])
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_src.dtype))
    return agg / (deg.unsqueeze(-1) + 1e-6)


class ResidualGatedLineGraphGNN(nn.Module):
    def __init__(self, d_in, hid=128, dropout=0.15, rounds=2):
        super().__init__()
        self.rounds = int(rounds)
        self.edge_enc = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(hid)
        self.local_head = nn.Linear(hid, 1)
        self.upd = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
        self.topo_head = nn.Linear(hid, 1)
        self.gate = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
            nn.Sigmoid()
        )

    def forward(self, X, edge_index, mp_mask):
        E = X.size(0)
        h0 = self.ln(self.edge_enc(X))
        logit_local = self.local_head(h0).squeeze(-1)

        agg0 = scatter_mean_edge(h0, edge_index, E)
        agg0 = agg0 * mp_mask.unsqueeze(-1)

        h = h0
        for _ in range(self.rounds):
            agg = scatter_mean_edge(h, edge_index, E)
            agg = agg * mp_mask.unsqueeze(-1)
            h = self.ln(h + 0.5 * self.upd(torch.cat([h, agg], dim=-1)))

        logit_topo = self.topo_head(h).squeeze(-1)
        g = self.gate(torch.cat([h0, agg0], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_local + g * logit_topo


@torch.no_grad()
def eval_gnn(model, cache, keys, pair_tensors=None, store_preds=False):
    model.eval()
    Ps, Ys, Pats, Ds = [], [], [], []
    for pk in keys:
        Xt, yt, edge_index, mp_mask = pair_tensors[pk]
        logits = model(Xt, edge_index, mp_mask)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = yt.detach().cpu().numpy().astype(int)

        Ps.append(p)
        Ys.append(y)
        Pats.append(np.array([cache[pk]["patient"]] * len(y), dtype=object))
        if store_preds:
            Ds.append(cache[pk]["dose_raw"].astype(np.float32))

    P = np.concatenate(Ps)
    Y = np.concatenate(Ys)
    PAT = np.concatenate(Pats)
    ma, mp, mia, mip, pr = eval_from_arrays(PAT, Y, P)

    if store_preds:
        DOSE = np.concatenate(Ds) if Ds else np.zeros((0,), np.float32)
        return ma, mp, mia, mip, pr, PAT, Y, P, DOSE

    return ma, mp, mia, mip, pr


def train_gnn(model, cache, train_keys, val_keys, pair_tensors, rep_seed: int):
    set_seed(rep_seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_NEURAL, weight_decay=WEIGHT_DECAY)
    crit = WeightedFocalBCE(gamma=FOCAL_GAMMA)
    posw = torch.tensor(compute_pos_weight_from_pairs(cache, train_keys), dtype=torch.float32, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    best_score = -1e18
    best_state = None
    wait = int(PATIENCE)
    train_keys_shuf = list(train_keys)

    for _ep in range(1, int(EPOCHS) + 1):
        model.train()
        random.shuffle(train_keys_shuf)

        for pk in train_keys_shuf:
            Xt, yt, edge_index, mp_mask = pair_tensors[pk]
            opt.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast(True):
                    logits = model(Xt, edge_index, mp_mask)
                    loss = crit(logits, yt, pos_weight=posw)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(Xt, edge_index, mp_mask)
                loss = crit(logits, yt, pos_weight=posw)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
                opt.step()

        v_macro_auc, v_macro_ap, _, _, _ = eval_gnn(model, cache, val_keys, pair_tensors=pair_tensors, store_preds=False)
        auc_term = (v_macro_auc if np.isfinite(v_macro_auc) else 0.0)
        ap_term = (v_macro_ap if np.isfinite(v_macro_ap) else 0.0)
        score = float(EARLYSTOP_W_AUC * auc_term + EARLYSTOP_W_AP * ap_term)

        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = int(PATIENCE)
        else:
            wait -= 1
            if wait <= 0:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def _dose_bins_for_cache(cache):
    all_doses = []
    for _, rec in cache.items():
        d = rec["dose_raw"]
        all_doses.append(d[np.isfinite(d)])
    all_doses = np.concatenate(all_doses) if all_doses else np.zeros((0,), np.float32)
    return _dose_bins_from_data(all_doses, width=DOSE_BIN_WIDTH_GY, min_bins=MIN_BINS)


def run_one(thr: str, modality: str):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_seed(SEED)

    for k in EDGE_NUM_KEYS_BASE + EDGE_BOOL_KEYS_BASE + EDGE_CAT_KEYS_BASE:
        if _is_leaky_key(k):
            raise ValueError(f"Leaky key in features: {k}")

    thr_dir = DAMAGE_ROOT / thr
    pairs = _collect_pairs_graphml(thr_dir, modality=modality)
    log(f"\n---THR={thr} | MOD={modality}---")
    log(f"Pairs pre-QC: {len(pairs)} | patients={pairs['patient'].nunique()}")

    pairs = apply_qc(pairs)
    log(f"Pairs post-QC: {len(pairs)} | patients={pairs['patient'].nunique()}")

    if HARD_EXCLUDE_PATIENTS:
        before_p = pairs["patient"].nunique()
        before_n = len(pairs)
        pairs = pairs[~pairs["patient"].isin(HARD_EXCLUDE_PATIENTS)].copy()
        log(f"[Hard exclude] removed_patients={before_p - pairs['patient'].nunique()} | removed_pairs={before_n - len(pairs)}")

    if pairs.empty or pairs["patient"].nunique() < (N_VAL_PATIENTS + TEST_GROUP_SIZE + 1):
        log("Not enough patients after QC/exclude.")
        return

    cache, by_pat = build_pair_cache(
        pairs,
        edge_num_keys_full=EDGE_NUM_KEYS_BASE,
        edge_bool_keys=EDGE_BOOL_KEYS_BASE,
        edge_cat_keys=EDGE_CAT_KEYS_BASE
    )

    pats = sorted(by_pat.keys())
    if len(pats) < (N_VAL_PATIENTS + TEST_GROUP_SIZE + 1):
        log("Not enough patients after caching.")
        return

    folds = make_patient_folds(pats, test_group_size=TEST_GROUP_SIZE, n_val=N_VAL_PATIENTS, seed=SEED)
    if LIMIT_FOLDS is not None:
        folds = folds[:int(LIMIT_FOLDS)]
    log(f"Folds: {len(folds)} | DEVICE={DEVICE} | CAP_LG_IN_DEG={CAP_LG_IN_DEG} | AMP={USE_AMP and DEVICE=='cuda'}")

    outdir = OUTBASE / thr / modality
    outdir.mkdir(parents=True, exist_ok=True)

    bin_edges = _dose_bins_for_cache(cache)

    trend_out = outdir / "dose_damage_trend.csv"
    trend = dose_damage_trend_from_cache(cache, trend_out, bin_edges=bin_edges)
    if trend is not None:
        _, corr = trend
        log(f"[Dose trend] saved dose_damage_trend.csv | corr(bin_median_dose, bin_damage_rate)={corr:.3f}")
    else:
        log("[Dose trend] insufficient data; skipped")

    rows = []
    preds_store = []

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    RUN_NAME = "full_no_film"

    for fold in folds:
        rep = int(fold["rep"])
        rep_seed = SEED + rep * 1337
        train_p, val_p, test_p = map(set, (fold["train"], fold["val"], fold["test"]))

        train_keys = sum([by_pat[p] for p in train_p], [])
        val_keys = sum([by_pat[p] for p in val_p], [])
        test_keys = sum([by_pat[p] for p in test_p], [])
        if not train_keys or not val_keys or not test_keys:
            continue

        mu_c_full, sd_c_full = fit_norm_cont_from_fullX(cache, train_keys)
        lobe_vocab = build_vocab_from_train(cache, train_keys, "lobe")
        side_vocab = build_vocab_from_train(cache, train_keys, "side")

        t_list = [cache[pk]["fuidx_raw"] for pk in train_keys]
        time_mu, time_sd = fit_norm_1d(t_list)

        pair_tensors = {}
        if PRECACHE_TENSORS_PER_FOLD:
            all_keys = list(dict.fromkeys(train_keys + val_keys + test_keys))
            for pk in all_keys:
                pair_tensors[pk] = materialize_pair_torch(
                    cache[pk],
                    mu_c_full, sd_c_full,
                    lobe_vocab, side_vocab,
                    time_mu, time_sd
                )

        Xt0, _, _, _ = pair_tensors[train_keys[0]]
        d_in = int(Xt0.shape[1])

        model = ResidualGatedLineGraphGNN(d_in=d_in, hid=HID, dropout=DROPOUT, rounds=ROUNDS)
        model = train_gnn(model, cache, train_keys, val_keys, pair_tensors, rep_seed=rep_seed)

        ma, mp, mia, mip, _, PAT, Y, P, DOSE = eval_gnn(
            model, cache, test_keys, pair_tensors=pair_tensors, store_preds=True
        )

        rows.append(dict(
            thr=thr,
            modality=modality,
            run=RUN_NAME,
            rep=rep,
            test_pats=",".join(sorted(test_p)),
            macro_auc=ma, macro_ap=mp,
            micro_auc=mia, micro_ap=mip
        ))
        preds_store.append((PAT, Y, P, DOSE))

        log(f"[{thr}|{modality}] rep={rep:02d} test={','.join(sorted(test_p))} "
            f"macroAUC={ma:.3f} macroAP={mp:.3f} microAUC={mia:.3f} microAP={mip:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        log("No results produced.")
        return

    safe_to_csv(df, outdir / "folds.csv")

    summ = (df.groupby("run")
              .agg(macro_auc_mean=("macro_auc", "mean"),
                   macro_auc_std=("macro_auc", "std"),
                   macro_ap_mean=("macro_ap", "mean"),
                   macro_ap_std=("macro_ap", "std"),
                   micro_auc_mean=("micro_auc", "mean"),
                   micro_ap_mean=("micro_ap", "mean"),
                   n=("rep", "count"))
              .reset_index()
              .sort_values("macro_auc_mean", ascending=False))
    safe_to_csv(summ, outdir / "summary.csv")

    log(f"\n[{thr}|{modality}] === SUMMARY ===")
    for r in summ.itertuples(index=False):
        log(f"{r.run:30s} | macroAUC={r.macro_auc_mean:.3f}±{(r.macro_auc_std if np.isfinite(r.macro_auc_std) else float('nan')):.3f} "
            f"| macroAP={r.macro_ap_mean:.3f}±{(r.macro_ap_std if np.isfinite(r.macro_ap_std) else float('nan')):.3f} "
            f"| microAUC={r.micro_auc_mean:.3f} microAP={r.micro_ap_mean:.3f} | n={int(r.n)}")

    PAT = np.concatenate([t[0] for t in preds_store])
    Y = np.concatenate([t[1] for t in preds_store])
    P = np.concatenate([t[2] for t in preds_store])
    DOSE = np.concatenate([t[3] for t in preds_store])

    out_cal = outdir / f"dose_calibration_{RUN_NAME}.csv"
    g = dose_calibration_from_predictions(PAT, Y, P, DOSE, out_cal, bin_edges=bin_edges)
    if g is not None:
        log(f"[Dose calibration] saved {out_cal}")
    else:
        log("[Dose calibration] insufficient preds; skipped")

    log(f"Saved → {outdir/'folds.csv'} {outdir/'summary.csv'} + dose files in {outdir}")


def main():
    set_seed(SEED)
    log(f"DEVICE={DEVICE} | AMP={USE_AMP and DEVICE=='cuda'} | CAP_LG_IN_DEG={CAP_LG_IN_DEG}")
    log(f"THRESHOLDS={THRESHOLDS} | MODALITIES={MODALITIES}")
    log(f"QC: ENABLE_QC={ENABLE_QC} | MATCH_FRAC_THRESHOLD={MATCH_FRAC_THRESHOLD} | BAD_GRAPHS_PATIENT_EXCLUDE={BAD_GRAPHS_PATIENT_EXCLUDE}")
    log(f"Caps: MAX_EDGES_PER_PAIR={MAX_EDGES_PER_PAIR} | MAX_EDGES_PER_PATIENT={MAX_EDGES_PER_PATIENT} | STRATIFIED_PAIR_SUBSAMPLE={STRATIFIED_PAIR_SUBSAMPLE}")
    log(f"Train: EPOCHS={EPOCHS} PATIENCE={PATIENCE} HID={HID} ROUNDS={ROUNDS} DROPOUT={DROPOUT}")

    for thr in THRESHOLDS:
        for mod in MODALITIES:
            run_one(thr, mod)

    log("\nDone.")


if __name__ == "__main__":
    main()
