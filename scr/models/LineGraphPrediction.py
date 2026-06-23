"""
Pulmonary vessel damage prediction using a line-graph GNN under LOPO cross-validation

This script:
- Loads labeled GraphML pairs per damage threshold and vessel modality
- Applies matching quality check
- Builds a cached, capped line-graph representation per pair once
- Runs LOPO-CV with shared patient-wise folds and trains two model configurations
  (FiLM conditioning vs. ablation without FiLM)
- Writes fold-level metrics, summary metrics, and dose-related analyses:
  - dose_damage_trend.csv (+ Spearman correlation)
  - Per-config dose calibration curves from held-out predictions
"""

import os, re, math, random, warnings, zlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Config — set DATA_ROOT via environment variable or edit the fallback path
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get(
    "VASCULAR_DATA_ROOT",
    r"C:\Users\ilinc\OneDrive\Desktop\GraphAnalysis\GraphsCompleteAnalysis\graphs_complete_cleaned"
))
flg         = DATA_ROOT / "final_labeled_graphs"
damage_root = flg / "damage_thresholds"

thresholds = ["damage_90pct", "damage_75pct", "damage_50pct"]
modalities = ["Artery", "Vein"]

seed   = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

cap_lg_in_deg           = 24
precache_tensors_per_fold = True
use_amp                 = True

batch_pairs_train  = 4
batch_pairs_eval   = 8
use_torch_compile  = True

epochs       = 50
patience     = 6
lr_neural    = 2e-3
weight_decay = 1e-4
grad_clip    = 2.0
dropout      = 0.15
hid          = 128
rounds       = 2
focal_gamma  = 1.0
focal_alpha  = 0.75          # fixed α as stated in paper (Section 2.4)
earlystop_w_auc = 0.5
earlystop_w_ap  = 0.5

film_scale = 0.10

test_group_size = 1
n_val_patients  = 3
limit_folds     = None

enable_qc              = True
match_frac_threshold   = 0.75
bad_graphs_patient_exclude = 2
report_csv = flg / "_postmatch_reports" / "lobe_counts_and_quality.csv"

hard_exclude_patients = {
    "P144", "P151", "P152", "P155", "P168",
    "P172", "P181", "P183", "P190", "P109"
}

max_edges_per_pair      = 15000
max_edges_per_patient   = 120000
stratified_pair_subsample        = True
keep_pos_fraction_when_cap       = 0.50

outbase = damage_root / "_final_full_all"
outbase.mkdir(parents=True, exist_ok=True)

dose_bin_width_gy = 5.0
min_bins          = 6

configs = [
    dict(name="C0_XdoseXtime__NoFiLM",         x_use_dose=True,  x_use_time=True,  film_mode="none"),
    dict(name="C3_NoDoseNoTime__FiLMdoseTime",  x_use_dose=False, x_use_time=False, film_mode="dose_time"),
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def safe_to_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    import builtins
    with builtins.open(path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)

def stable_pair_seed(pair_key: str, base_seed: int = seed) -> int:
    h = zlib.crc32(pair_key.encode("utf-8")) & 0xFFFFFFFF
    return int((base_seed + h) % (2**31 - 1))

# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

leak_substr = ("label", "final_label", "damage", "damaged", "disappear", "surviv")

def _is_leaky_key(k: str) -> bool:
    s = str(k).strip().lower()
    return any(t in s for t in leak_substr)

# ---------------------------------------------------------------------------
# Timepoint parser
# ---------------------------------------------------------------------------

tp_re = re.compile(
    r"FU\s*0*(\d+)"
    r"(?:\s*[-_ ]?\s*(?:M|m|mo|month|months)?\s*0*([\d]+(?:\.[\d]+)?))?"
    r"\s*$",
    re.IGNORECASE
)

def _parse_tp(tp: str):
    m = tp_re.match(str(tp))
    if not m:
        return (np.nan, np.nan)
    fu_idx = float(m.group(1)) if m.group(1) else np.nan
    fu_mon = float(m.group(2)) if (m.group(2) not in (None, "")) else np.nan
    return fu_idx, fu_mon

# ---------------------------------------------------------------------------
# Feature key lists — validated at import time
# ---------------------------------------------------------------------------

edge_num_keys_with_dose = [
    "length", "tortuosity", "radius_avg", "radius_min", "radius_max",
    "radius_SD", "vis_radius", "surface_area", "volume", "dose_gy",
    "patient_tumor_volume_ml", "log1p_tumor_volume_ml", "patient_tumor_voxels",
]
edge_num_keys_no_dose = [
    "length", "tortuosity", "radius_avg", "radius_min", "radius_max",
    "radius_SD", "vis_radius", "surface_area", "volume",
    "patient_tumor_volume_ml", "log1p_tumor_volume_ml", "patient_tumor_voxels",
]
edge_bool_keys = ["is_ipsilateral", "is_in_tumor_lobe"]
edge_cat_keys  = ["lobe", "side"]

for _k in edge_num_keys_with_dose + edge_bool_keys + edge_cat_keys:
    if _is_leaky_key(_k):
        raise ValueError(f"Leaky key in feature list: {_k}")

# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

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
    if s in ("1", "true", "yes"):  return 1.0
    if s in ("0", "false", "no"):  return 0.0
    return np.nan

def _parse_final_label_to_y(final_label):
    s = "" if final_label is None else str(final_label).strip().lower()
    if s in ("damaged", "disappeared"): return 1.0
    if s in ("survived",):              return 0.0
    return np.nan

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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

def eval_from_arrays(pat, y, p):
    micro_auc, micro_ap = _metrics(y, p)
    macro_auc, macro_ap = macro_auc_ap_per_patient(pat, y, p)
    pos_rate = float(np.mean(y)) if len(y) else float("nan")
    return macro_auc, macro_ap, micro_auc, micro_ap, pos_rate

# ---------------------------------------------------------------------------
# QC / data loading
# ---------------------------------------------------------------------------

def _read_match_report(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["patient", "modality", "timepoint", "match_frac"])
    rep  = pd.read_csv(csv_path)
    cols = list(rep.columns)

    def _find(cands):
        for c in cands:
            if c in cols: return c
        return None

    pcol = _find(["patient", "pid", "Patient"])
    mcol = _find(["modality", "mod", "Modality"])
    tcol = _find(["timepoint", "tp", "Timepoint", "FU"])
    fcol = _find(["match_frac", "match_fraction", "matching_fraction", "quality", "match"])

    if pcol: rep = rep.rename(columns={pcol: "patient"})
    if mcol: rep = rep.rename(columns={mcol: "modality"})
    if tcol: rep = rep.rename(columns={tcol: "timepoint"})
    if fcol: rep = rep.rename(columns={fcol: "match_frac"})
    if "match_frac" not in rep.columns:
        rep["match_frac"] = np.nan

    rep["patient"]    = rep.get("patient",   "").astype(str).str.strip()
    rep["modality"]   = rep.get("modality",  "").astype(str).str.strip()
    rep["timepoint"]  = rep.get("timepoint", "").astype(str).str.strip()
    rep["match_frac"] = pd.to_numeric(rep["match_frac"], errors="coerce")
    return rep[["patient", "modality", "timepoint", "match_frac"]]

def _collect_pairs_graphml(thr_dir: Path, modality: str) -> pd.DataFrame:
    rows = []
    for pdir in sorted([d for d in thr_dir.glob("P*/") if d.is_dir()]):
        pat  = pdir.name
        mdir = pdir / modality
        if not mdir.is_dir():
            continue
        for fu in sorted([d for d in mdir.iterdir()
                          if d.is_dir() and d.name.upper().startswith("FU")]):
            gpath = fu / "BL_labeled.graphml"
            if gpath.exists():
                rows.append({
                    "patient":   pat,
                    "modality":  modality,
                    "timepoint": fu.name,
                    "graphml":   str(gpath),
                })
    return pd.DataFrame(rows)

def apply_qc(pairs: pd.DataFrame) -> pd.DataFrame:
    repdf = _read_match_report(report_csv)
    pairs = pairs.merge(repdf, how="left", on=["patient", "modality", "timepoint"])
    if not enable_qc:
        return pairs

    bad        = pairs[pairs["match_frac"].notna() & (pairs["match_frac"] < match_frac_threshold)].copy()
    bad_counts = bad.groupby("patient").size().to_dict()
    exclude_patients = sorted([p for p, c in bad_counts.items() if c >= bad_graphs_patient_exclude])

    keep = pairs.copy()
    keep = keep[~keep["patient"].isin(exclude_patients)].copy()
    keep = keep[keep["match_frac"].isna() | (keep["match_frac"] >= match_frac_threshold)].copy()

    log(f"[QC] removed_pairs={len(pairs)-len(keep)} | "
        f"removed_patients={pairs['patient'].nunique()-keep['patient'].nunique()}")
    return keep

# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def make_patient_folds(patients_sorted, test_group_size=1, n_val=3, seed=seed):
    pats = list(patients_sorted)
    if len(pats) < (n_val + test_group_size + 1):
        raise ValueError(f"Need at least {n_val + test_group_size + 1} patients, got {len(pats)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)

    folds   = []
    n_folds = int(math.ceil(len(pats) / float(test_group_size)))
    for i in range(n_folds):
        test = pats[i * test_group_size:(i + 1) * test_group_size]
        if not test:
            continue
        remain = [p for p in pats if p not in test]
        rng_i  = np.random.default_rng(seed + 1337 * (i + 1))
        rng_i.shuffle(remain)
        val   = remain[:n_val]
        train = [p for p in remain if p not in val]
        folds.append({"rep": i, "train": train, "val": val, "test": test})

    all_test = sum([r["test"] for r in folds], [])
    assert len(all_test) == len(set(all_test)) == len(pats)
    return folds

# ---------------------------------------------------------------------------
# Edge subsampling
# ---------------------------------------------------------------------------

def _stratified_cap_indices(y, max_n, seed):
    y   = y.astype(int)
    idx = np.arange(len(y), dtype=np.int64)
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    pos = idx[y == 1]
    neg = idx[y == 0]

    max_pos = int(round(max_n * float(keep_pos_fraction_when_cap)))
    n_pos   = min(len(pos), max_pos)
    n_neg   = max_n - n_pos

    keep_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=np.int64)
    keep_neg = neg if len(neg) <= n_neg else rng.choice(neg, size=n_neg, replace=False)

    keep = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep)
    keep.sort()
    return keep

# ---------------------------------------------------------------------------
# Line-graph construction
# ---------------------------------------------------------------------------

def build_linegraph_edge_index_none_capped(u, v, n_nodes, cap_k=None, seed0=0):
    inc = [[] for _ in range(int(n_nodes))]
    for ei, (uu, vv) in enumerate(zip(u, v)):
        inc[int(uu)].append(ei)
        inc[int(vv)].append(ei)

    rng = np.random.default_rng(seed0)
    src, dst = [], []

    for lst in inc:
        L = len(lst)
        if L < 2:
            continue
        if cap_k is not None and L > int(cap_k):
            lst = rng.choice(np.asarray(lst, np.int64), size=int(cap_k), replace=False).tolist()
            L   = len(lst)
        for i in range(L):
            a = lst[i]
            for j in range(L):
                if i == j:
                    continue
                src.append(a)
                dst.append(lst[j])

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.asarray(src, np.int64), np.asarray(dst, np.int64)], axis=0)

# ---------------------------------------------------------------------------
# GraphML → pair cache
# ---------------------------------------------------------------------------

def graphml_to_pair_cache(graphml_path: str, patient: str, timepoint: str):
    G      = nx.read_graphml(graphml_path)
    fu_idx, _ = _parse_tp(timepoint)

    node_ids = list(G.nodes())
    nid2i    = {str(n): i for i, n in enumerate(node_ids)}

    u_idx, v_idx        = [], []
    X_with_dose, X_no_dose, X_bool = [], [], []
    e_cat_raw           = {k: [] for k in edge_cat_keys}
    y_list              = []
    dose_raw, fuidx_raw = [], []

    for u, v, ed in G.edges(data=True):
        y = _parse_final_label_to_y(ed.get("final_label", None))
        if not np.isfinite(y):
            continue

        vals_with = [_as_float(ed.get(k, np.nan)) for k in edge_num_keys_with_dose]
        vals_no   = [_as_float(ed.get(k, np.nan)) for k in edge_num_keys_no_dose]
        bvals     = [_as_bool01(ed.get(k, np.nan)) for k in edge_bool_keys]

        X_with_dose.append(vals_with)
        X_no_dose.append(vals_no)
        X_bool.append(bvals)
        y_list.append(float(y))

        for k in edge_cat_keys:
            e_cat_raw[k].append("" if ed.get(k, None) is None else str(ed.get(k)))

        u_idx.append(nid2i[str(u)])
        v_idx.append(nid2i[str(v)])
        dose_raw.append(_as_float(ed.get("dose_gy", np.nan)))
        fuidx_raw.append(float(fu_idx) if np.isfinite(fu_idx) else np.nan)

    if len(y_list) < 5:
        return None

    X_with_dose = np.asarray(X_with_dose, np.float32)
    X_no_dose   = np.asarray(X_no_dose,   np.float32)
    X_bool      = np.asarray(X_bool,      np.float32)
    y           = np.asarray(y_list,      np.float32)
    u_idx       = np.asarray(u_idx,       np.int64)
    v_idx       = np.asarray(v_idx,       np.int64)
    n_nodes     = len(node_ids)
    dose_raw    = np.asarray(dose_raw,    np.float32)
    fuidx_raw   = np.asarray(fuidx_raw,  np.float32)

    pair_key = f"{patient}|{timepoint}"
    ps       = stable_pair_seed(pair_key, seed)

    idx = np.arange(len(y), dtype=np.int64)
    if max_edges_per_pair is not None and idx.size > int(max_edges_per_pair):
        if stratified_pair_subsample:
            idx = _stratified_cap_indices(y, int(max_edges_per_pair), seed=ps)
        else:
            rng = np.random.default_rng(ps)
            idx = rng.choice(idx, size=int(max_edges_per_pair), replace=False)
            idx.sort()

        X_with_dose = X_with_dose[idx]
        X_no_dose   = X_no_dose[idx]
        X_bool      = X_bool[idx]
        y           = y[idx]
        u_idx       = u_idx[idx]
        v_idx       = v_idx[idx]
        dose_raw    = dose_raw[idx]
        fuidx_raw   = fuidx_raw[idx]
        for k in edge_cat_keys:
            e_cat_raw[k] = [e_cat_raw[k][i] for i in idx.tolist()]

    edge_index_none = build_linegraph_edge_index_none_capped(
        u_idx, v_idx, n_nodes, cap_k=cap_lg_in_deg, seed0=ps
    )

    return dict(
        patient=patient,
        timepoint=timepoint,
        pair_key=pair_key,
        X_with_dose=X_with_dose,
        X_no_dose=X_no_dose,
        X_bool=X_bool,
        y=y,
        edge_cat_raw={k: np.asarray(vv, object) for k, vv in e_cat_raw.items()},
        dose_raw=dose_raw,
        fuidx_raw=fuidx_raw,
        edge_index_none=edge_index_none,
    )

def build_pair_cache(pairs: pd.DataFrame):
    cache    = {}
    by_pat   = defaultdict(list)
    pat_edge_counts = defaultdict(int)
    skipped  = 0

    for r in pairs.itertuples(index=False):
        rec = graphml_to_pair_cache(r.graphml, r.patient, r.timepoint)
        if rec is None:
            skipped += 1
            continue

        nE = len(rec["y"])
        if max_edges_per_patient is not None:
            if pat_edge_counts[r.patient] + nE > int(max_edges_per_patient):
                continue
            pat_edge_counts[r.patient] += nE

        cache[rec["pair_key"]] = rec
        by_pat[r.patient].append(rec["pair_key"])

    log(f"[cache] patients={len(by_pat)} | pairs={len(cache)} | skipped_pairs={skipped}")
    return cache, by_pat

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def fit_norm_cont(cache, pair_keys, which="with_dose"):
    sumv = sumsq = cnt = None
    for pk in pair_keys:
        X  = cache[pk]["X_with_dose" if which == "with_dose" else "X_no_dose"].astype(np.float32)
        m  = np.isfinite(X)
        X0 = np.where(m, X, 0.0)
        if sumv is None:
            d    = X.shape[1]
            sumv = np.zeros((d,), np.float64)
            sumsq = np.zeros((d,), np.float64)
            cnt  = np.zeros((d,), np.float64)
        sumv  += X0.sum(axis=0)
        sumsq += (X0 * X0).sum(axis=0)
        cnt   += m.sum(axis=0)

    cnt = np.maximum(cnt, 1.0)
    mu  = (sumv / cnt).astype(np.float32)
    var = (sumsq / cnt) - (mu.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-6)
    sd  = np.sqrt(var).astype(np.float32)
    sd  = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32)
    mu  = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    return mu, sd

def fit_norm_1d(values_list):
    x  = np.concatenate(values_list, axis=0) if values_list else np.zeros((0,), np.float32)
    mu = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    sd = float(np.nanstd(x))  if np.isfinite(x).any() else 1.0
    if not np.isfinite(sd) or sd < 1e-6: sd = 1.0
    if not np.isfinite(mu):              mu = 0.0
    return np.float32(mu), np.float32(sd)

def build_vocab_from_train(cache, pair_keys, field):
    vocab = {"": 0}
    for pk in pair_keys:
        for v in cache[pk]["edge_cat_raw"][field]:
            s = "" if v is None else str(v)
            if s not in vocab:
                vocab[s] = len(vocab)
    return vocab

# ---------------------------------------------------------------------------
# Dose analysis helpers
# ---------------------------------------------------------------------------

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
    ys, ds = [], []
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
    df["dose_bin"] = pd.cut(df["dose_gy"], bins=bin_edges, include_lowest=True).astype(str)

    g = (df.groupby("dose_bin", observed=True)
           .agg(n=("y", "size"),
                damage_rate=("y", "mean"),
                dose_median=("dose_gy", "median"),
                dose_mean=("dose_gy", "mean"))
           .reset_index()
           .sort_values("dose_median"))

    safe_to_csv(g, out_csv)

    if len(g) >= 3:
        rho, _p = spearmanr(g["dose_median"].values, g["damage_rate"].values, nan_policy="omit")
        corr = float(rho) if np.isfinite(rho) else float("nan")
    else:
        corr = float("nan")

    return g, corr

def dose_calibration_from_predictions(pat, y, p, dose, out_csv: Path, bin_edges):
    df = pd.DataFrame({
        "patient": np.asarray(pat, object),
        "y":       np.asarray(y,   int),
        "p":       np.asarray(p,   float),
        "dose_gy": np.asarray(dose, float),
    })
    df = df[np.isfinite(df["dose_gy"].values)].copy()
    if len(df) < 50:
        return None

    df["dose_bin"] = pd.cut(df["dose_gy"], bins=bin_edges, include_lowest=True).astype(str)

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

# ---------------------------------------------------------------------------
# Feature materialisation (per fold, per config)
# ---------------------------------------------------------------------------

def materialize_pair(rec, cfg,
                     mu_cont, sd_cont,
                     lobe_vocab, side_vocab,
                     mu_time, sd_time,
                     mu_dose, sd_dose):
    Xc0 = rec["X_with_dose"] if cfg["x_use_dose"] else rec["X_no_dose"]
    Xc  = (Xc0.astype(np.float32) - mu_cont) / sd_cont
    Xc  = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)

    Xb    = np.nan_to_num(rec["X_bool"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    parts = [Xc, Xb]

    if cfg["x_use_time"]:
        t    = rec["fuidx_raw"].astype(np.float32)
        tt   = (t - float(mu_time)) / float(sd_time)
        tok_t = np.isfinite(t).astype(np.float32)
        tt   = np.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append((tt * tok_t)[:, None].astype(np.float32))

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
    mp_mask       = np.ones((len(y),), np.float32)

    film_mode = cfg["film_mode"]
    T = Tok = None
    if film_mode != "none":
        dose = rec["dose_raw"].astype(np.float32)
        time = rec["fuidx_raw"].astype(np.float32)

        td = (dose - float(mu_dose)) / float(sd_dose)
        td = np.nan_to_num(td, nan=0.0, posinf=0.0, neginf=0.0)

        if film_mode == "dose":
            Tok = np.isfinite(dose).astype(np.float32)
            T   = td[:, None].astype(np.float32)
        else:  # dose_time
            tt  = (time - float(mu_time)) / float(sd_time)
            tt  = np.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)
            Tok = (np.isfinite(dose) & np.isfinite(time)).astype(np.float32)
            T   = np.stack([td, tt], axis=1).astype(np.float32)

    Xt       = torch.from_numpy(X).float().to(device)
    yt       = torch.from_numpy(y).float().to(device)
    edge_idx = torch.from_numpy(edge_index_np).long().to(device)
    mp_mask_t = torch.from_numpy(mp_mask).float().to(device)

    if T is None:
        return (Xt, yt, edge_idx, mp_mask_t, None, None)
    return (Xt, yt, edge_idx, mp_mask_t,
            torch.from_numpy(T).float().to(device),
            torch.from_numpy(Tok).float().to(device))

# ---------------------------------------------------------------------------
# Loss — Fixed α=0.75 as stated in paper (Section 2.4)
# ---------------------------------------------------------------------------

class WeightedFocalBCE(nn.Module):
    """
    Focal loss with:
      - fixed α (alpha) balancing positive vs. negative class (paper: α=0.75)
      - per-fold dynamic pos_weight as an additional minority scaling factor
      - γ (gamma) focusing parameter (paper: γ=1.0)
    """
    def __init__(self, gamma: float = 0.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, y: torch.Tensor,
                pos_weight: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

        # α weighting (fixed, per paper)
        alpha_t = torch.where(y > 0.5,
                              torch.full_like(y, self.alpha),
                              torch.full_like(y, 1.0 - self.alpha))

        # dynamic per-fold class-ratio weight
        pw = torch.where(y > 0.5, pos_weight, torch.ones_like(y))

        if self.gamma <= 0:
            return (alpha_t * pw * bce).mean()

        p      = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        pt     = p * y + (1.0 - p) * (1.0 - y)
        focal  = (1.0 - pt).pow(self.gamma)
        return (alpha_t * pw * focal * bce).mean()


def compute_pos_weight_from_pairs(cache, pair_keys):
    ys    = [cache[pk]["y"].astype(np.float32) for pk in pair_keys]
    y     = np.concatenate(ys).astype(int) if ys else np.array([], int)
    if y.size == 0:
        return 1.0
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(np.clip(n_neg / max(1, n_pos), 1.0, 12.0))

# ---------------------------------------------------------------------------
# Message passing helpers
# ---------------------------------------------------------------------------

def scatter_mean_edge(h_src: torch.Tensor, edge_index: torch.Tensor, E: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((E, h_src.size(1)), device=h_src.device, dtype=h_src.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    H   = h_src.size(1)
    agg = torch.zeros((E, H), device=h_src.device, dtype=h_src.dtype)
    deg = torch.zeros((E,),    device=h_src.device, dtype=h_src.dtype)
    agg.index_add_(0, dst, h_src[src])
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h_src.dtype))
    return agg / (deg.unsqueeze(-1) + 1e-6)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualGatedLineGraphGNN_FlexibleFiLM(nn.Module):
    """
    Residual gated line-graph GNN with optional FiLM conditioning.
      - hid=128, rounds=2 (paper Section 2.3)
      - FiLM scale=0.10, applied after each round when film_mode != 'none'
      - Gated blend of local and topology-aware logits
    """
    def __init__(self, d_in: int, hid: int = 128, dropout: float = 0.15,
                 rounds: int = 2, film_scale: float = 0.10, t_dim: int = 0):
        super().__init__()
        self.rounds     = int(rounds)
        self.film_scale = float(film_scale)
        self.t_dim      = int(t_dim)

        self.edge_enc = nn.Sequential(
            nn.Linear(d_in, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, hid),  nn.GELU(),
        )
        self.ln          = nn.LayerNorm(hid)
        self.local_head  = nn.Linear(hid, 1)

        self.upd = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, hid),     nn.GELU(),
        )
        self.topo_head = nn.Linear(hid, 1)

        self.gate = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.GELU(),
            nn.Linear(hid, 1),       nn.Sigmoid(),
        )

        if self.t_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(self.t_dim, hid), nn.GELU(),
                nn.Linear(hid, 2 * hid),
            )
        else:
            self.film = None

    def _apply_film(self, h: torch.Tensor,
                    T: torch.Tensor, Tok: torch.Tensor) -> torch.Tensor:
        gb   = self.film(T)
        g, b = gb.chunk(2, dim=-1)
        g    = torch.tanh(g)
        h2   = h * (1.0 + self.film_scale * g) + self.film_scale * b
        return h + Tok.unsqueeze(-1) * (h2 - h)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor,
                mp_mask: torch.Tensor,
                T: torch.Tensor = None, Tok: torch.Tensor = None) -> torch.Tensor:
        E  = X.size(0)
        h0 = self.ln(self.edge_enc(X))
        if self.film is not None and T is not None and Tok is not None:
            h0 = self.ln(self._apply_film(h0, T, Tok))

        logit_local = self.local_head(h0).squeeze(-1)

        agg0 = scatter_mean_edge(h0, edge_index, E) * mp_mask.unsqueeze(-1)

        h = h0
        for _ in range(self.rounds):
            agg = scatter_mean_edge(h, edge_index, E) * mp_mask.unsqueeze(-1)
            h   = self.ln(h + 0.5 * self.upd(torch.cat([h, agg], dim=-1)))
            if self.film is not None and T is not None and Tok is not None:
                h = self.ln(self._apply_film(h, T, Tok))

        logit_topo = self.topo_head(h).squeeze(-1)
        g          = self.gate(torch.cat([h0, agg0], dim=-1)).squeeze(-1)
        return (1.0 - g) * logit_local + g * logit_topo

# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

def collate_pairs(pair_tensors, keys):
    Xs, Ys, Ms, eis   = [], [], [], []
    Ts, Toks          = [], []
    off               = 0

    for pk in keys:
        Xt, yt, ei, mp, T, Tok = pair_tensors[pk]
        E = Xt.size(0)
        Xs.append(Xt); Ys.append(yt); Ms.append(mp)
        eis.append(ei + off)
        off += E
        if T is not None:
            Ts.append(T); Toks.append(Tok)

    X          = torch.cat(Xs, 0)
    y          = torch.cat(Ys, 0)
    mp         = torch.cat(Ms, 0)
    edge_index = torch.cat(eis, 1) if eis else torch.zeros((2, 0), device=X.device, dtype=torch.long)
    T_out      = torch.cat(Ts,   0) if Ts   else None
    Tok_out    = torch.cat(Toks, 0) if Toks else None
    return X, y, edge_index, mp, T_out, Tok_out

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_gnn(model, cache, keys, pair_tensors, store_preds=False):
    model.eval()
    Ps, Ys, Pats, Ds = [], [], [], []

    for pk in keys:
        Xt, yt, edge_index, mp_mask, T, Tok = pair_tensors[pk]
        logits = model(Xt, edge_index, mp_mask, T=T, Tok=Tok)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = yt.detach().cpu().numpy().astype(int)
        Ps.append(p); Ys.append(y)
        Pats.append(np.array([cache[pk]["patient"]] * len(y), dtype=object))
        if store_preds:
            Ds.append(cache[pk]["dose_raw"].astype(np.float32))

    P   = np.concatenate(Ps)
    Y   = np.concatenate(Ys)
    PAT = np.concatenate(Pats)
    ma, mp_metric, mia, mip, pr = eval_from_arrays(PAT, Y, P)

    if store_preds:
        DOSE = np.concatenate(Ds) if Ds else np.zeros((0,), np.float32)
        return ma, mp_metric, mia, mip, pr, PAT, Y, P, DOSE
    return ma, mp_metric, mia, mip, pr

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gnn(model, cache, train_keys, val_keys, pair_tensors, rep_seed: int):
    set_seed(rep_seed)
    model = model.to(device)

    if use_torch_compile and device == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            log(f"[torch.compile] skipped ({type(e).__name__}: {e})")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr_neural, weight_decay=weight_decay)
    # α=0.75 and γ=1.0 as stated in paper (Section 2.4)
    crit  = WeightedFocalBCE(gamma=focal_gamma, alpha=focal_alpha)
    posw  = torch.tensor(
        compute_pos_weight_from_pairs(cache, train_keys),
        dtype=torch.float32, device=device
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_score = -1e18
    best_state = None
    wait       = int(patience)
    train_keys_shuf = list(train_keys)

    for _ep in range(1, int(epochs) + 1):
        model.train()
        random.shuffle(train_keys_shuf)

        bs = int(max(1, batch_pairs_train))
        for i in range(0, len(train_keys_shuf), bs):
            bkeys = train_keys_shuf[i:i + bs]
            if not bkeys:
                continue

            X, y, edge_index, mp_mask, T, Tok = collate_pairs(pair_tensors, bkeys)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast(True):
                    logits = model(X, edge_index, mp_mask, T=T, Tok=Tok)
                    loss   = crit(logits, y, pos_weight=posw)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X, edge_index, mp_mask, T=T, Tok=Tok)
                loss   = crit(logits, y, pos_weight=posw)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()

        v_macro_auc, v_macro_ap, _, _, _ = eval_gnn(
            model, cache, val_keys, pair_tensors=pair_tensors, store_preds=False
        )
        auc_term = v_macro_auc if np.isfinite(v_macro_auc) else 0.0
        ap_term  = v_macro_ap  if np.isfinite(v_macro_ap)  else 0.0
        score    = float(earlystop_w_auc * auc_term + earlystop_w_ap * ap_term)

        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait       = int(patience)
        else:
            wait -= 1
            if wait <= 0:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

# ---------------------------------------------------------------------------
# Main per-threshold / per-modality runner
# ---------------------------------------------------------------------------

def run_all_configs_for_thr_mod(thr: str, modality: str):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_seed(seed)

    thr_dir = damage_root / thr
    pairs   = _collect_pairs_graphml(thr_dir, modality=modality)
    log(f"\n[thr={thr} | mod={modality}] pairs pre-QC: {len(pairs)} | "
        f"patients={pairs['patient'].nunique()}")

    pairs = apply_qc(pairs)
    log(f"[thr={thr} | mod={modality}] pairs post-QC: {len(pairs)} | "
        f"patients={pairs['patient'].nunique()}")

    if hard_exclude_patients:
        before_p = pairs["patient"].nunique()
        before_n = len(pairs)
        pairs = pairs[~pairs["patient"].isin(hard_exclude_patients)].copy()
        log(f"[hard exclude] removed_patients={before_p - pairs['patient'].nunique()} | "
            f"removed_pairs={before_n - len(pairs)}")

    if pairs.empty or pairs["patient"].nunique() < (n_val_patients + test_group_size + 1):
        log("Not enough patients after QC/exclude — skipping.")
        return

    cache, by_pat = build_pair_cache(pairs)
    pats          = sorted(by_pat.keys())
    if len(pats) < (n_val_patients + test_group_size + 1):
        log("Not enough patients after caching — skipping.")
        return

    folds = make_patient_folds(pats, test_group_size=test_group_size,
                               n_val=n_val_patients, seed=seed)
    if limit_folds is not None:
        folds = folds[:int(limit_folds)]

    outdir_base = outbase / thr / modality
    outdir_base.mkdir(parents=True, exist_ok=True)

    folds_df = pd.DataFrame([{
        "rep":        f["rep"],
        "train_pats": ",".join(f["train"]),
        "val_pats":   ",".join(f["val"]),
        "test_pats":  ",".join(f["test"]),
    } for f in folds])
    safe_to_csv(folds_df, outdir_base / "cv_folds_patientwise_SHARED.csv")

    # Dose bins from all cached pairs
    all_doses = np.concatenate(
        [rec["dose_raw"][np.isfinite(rec["dose_raw"])] for rec in cache.values()]
    ) if cache else np.zeros((0,), np.float32)
    bin_edges = _dose_bins_from_data(all_doses, width=dose_bin_width_gy, min_bins=min_bins)

    trend = dose_damage_trend_from_cache(cache, outdir_base / "dose_damage_trend.csv", bin_edges)
    if trend is not None:
        _, corr = trend
        log(f"[dose trend] saved dose_damage_trend.csv | corr={corr:.3f}")
        safe_to_csv(
            pd.DataFrame([{"corr_bin_median_dose_vs_damage_rate": corr}]),
            outdir_base / "dose_damage_trend_corr.csv"
        )
    else:
        log("[dose trend] insufficient data — skipped.")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True

    for cfg in configs:
        cfg_name = cfg["name"]
        log(f"\n[running] {thr} | {modality} | {cfg_name}")

        outdir = outdir_base / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        rows        = []
        preds_store = []

        for fold in folds:
            rep      = int(fold["rep"])
            rep_seed = seed + rep * 1337

            train_p, val_p, test_p = map(set, (fold["train"], fold["val"], fold["test"]))
            train_keys = sum([by_pat[p] for p in train_p], [])
            val_keys   = sum([by_pat[p] for p in val_p],   [])
            test_keys  = sum([by_pat[p] for p in test_p],  [])
            if not train_keys or not val_keys or not test_keys:
                continue

            which = "with_dose" if cfg["x_use_dose"] else "no_dose"
            mu_cont, sd_cont = fit_norm_cont(cache, train_keys, which=which)

            lobe_vocab = build_vocab_from_train(cache, train_keys, "lobe")
            side_vocab = build_vocab_from_train(cache, train_keys, "side")

            mu_time, sd_time = fit_norm_1d([cache[pk]["fuidx_raw"] for pk in train_keys])
            mu_dose, sd_dose = fit_norm_1d([cache[pk]["dose_raw"]  for pk in train_keys])

            all_keys      = list(dict.fromkeys(train_keys + val_keys + test_keys))
            pair_tensors  = {}
            if precache_tensors_per_fold:
                for pk in all_keys:
                    pair_tensors[pk] = materialize_pair(
                        cache[pk], cfg,
                        mu_cont, sd_cont,
                        lobe_vocab, side_vocab,
                        mu_time, sd_time,
                        mu_dose, sd_dose,
                    )

            Xt0, _, _, _, _, _ = pair_tensors[train_keys[0]]
            d_in  = int(Xt0.shape[1])
            t_dim = 0 if cfg["film_mode"] == "none" else (1 if cfg["film_mode"] == "dose" else 2)

            model = ResidualGatedLineGraphGNN_FlexibleFiLM(
                d_in=d_in, hid=hid, dropout=dropout, rounds=rounds,
                film_scale=film_scale, t_dim=t_dim,
            )
            model = train_gnn(model, cache, train_keys, val_keys, pair_tensors,
                              rep_seed=rep_seed)

            ma, mp_metric, mia, mip, _, PAT, Y, P, DOSE = eval_gnn(
                model, cache, test_keys, pair_tensors=pair_tensors, store_preds=True
            )

            rows.append(dict(
                thr=thr, modality=modality, config=cfg_name, rep=rep,
                test_pats=",".join(sorted(test_p)),
                macro_auc=ma, macro_ap=mp_metric, micro_auc=mia, micro_ap=mip,
            ))
            preds_store.append((PAT, Y, P, DOSE))

            log(f"[{thr}|{modality}|{cfg_name}] rep={rep:02d} "
                f"test={','.join(sorted(test_p))} "
                f"macroAUC={ma:.3f} macroAP={mp_metric:.3f} "
                f"microAUC={mia:.3f} microAP={mip:.3f}")

        df = pd.DataFrame(rows)
        if df.empty:
            log(f"[{cfg_name}] no results produced.")
            continue

        safe_to_csv(df, outdir / "folds.csv")

        summ = (df.groupby("config")
                  .agg(macro_auc_mean=("macro_auc", "mean"),
                       macro_auc_std=("macro_auc", "std"),
                       macro_ap_mean=("macro_ap",  "mean"),
                       macro_ap_std=("macro_ap",  "std"),
                       micro_auc_mean=("micro_auc", "mean"),
                       micro_ap_mean=("micro_ap",  "mean"),
                       n=("rep", "count"))
                  .reset_index()
                  .sort_values("macro_auc_mean", ascending=False))
        safe_to_csv(summ, outdir / "summary.csv")

        PAT_all  = np.concatenate([t[0] for t in preds_store])
        Y_all    = np.concatenate([t[1] for t in preds_store])
        P_all    = np.concatenate([t[2] for t in preds_store])
        DOSE_all = np.concatenate([t[3] for t in preds_store])

        g_cal = dose_calibration_from_predictions(
            PAT_all, Y_all, P_all, DOSE_all,
            outdir / f"dose_calibration_{cfg_name}.csv",
            bin_edges=bin_edges,
        )
        if g_cal is None:
            log(f"[{cfg_name}] dose calibration skipped (insufficient predictions).")
        else:
            log(f"[{cfg_name}] dose calibration saved.")

        log(f"[{cfg_name}] saved → {outdir / 'folds.csv'} | {outdir / 'summary.csv'}")

    log(f"\n[{thr}|{modality}] done.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    set_seed(seed)
    log(f"device={device} | amp={use_amp and device == 'cuda'} | "
        f"cap_lg_in_deg={cap_lg_in_deg} | film_scale={film_scale}")
    log(f"focal: alpha={focal_alpha} gamma={focal_gamma}")
    log(f"batch_pairs_train={batch_pairs_train} | torch.compile={use_torch_compile}")
    log(f"thresholds={thresholds} | modalities={modalities}")
    log("configs:")
    for c in configs:
        log(f"  - {c['name']}: x_use_dose={c['x_use_dose']} "
            f"x_use_time={c['x_use_time']} film_mode={c['film_mode']}")

    for thr in thresholds:
        for mod in modalities:
            run_all_configs_for_thr_mod(thr, mod)

    log("\nDone.")


if __name__ == "__main__":
    main()
