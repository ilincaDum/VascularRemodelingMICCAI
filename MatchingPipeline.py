"""
Matching Algorithm Overview 

1. Read graphs and extract geometry
2. Global alignment (Umeyama)
3. Stage 1 - matching per lobe (Linear Sum Assignment )
4. Stage 2 - rescue matching for leftovers (mutual nearest neighbor)
5. Damage detection 

"""

import os, re, math, shutil
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



#configs

VALID_MODALITIES_DEFAULT = ["Artery", "Vein"]
SHOW_INLINE_FIRST_N_DEFAULT = 6

CENTRAL_LABELS = {"0", "central", "Central"}
FAST_DTYPE = np.float32
FAST_K_NEIGHBORS = 48

USE_PASS0_SIMILARITY = True
PASS0_MIN_ANCHORS = 100
PASS0_RANDOM_SUBSAMPLE = 2000

USE_FEATURED_LSA = True
FEAT_COST_WEIGHT = 0.30

RESCUE_SEARCH_RADIUS_MM = 15.0
RESCUE_LENGTH_TOLERANCE_FRAC = 0.75
RESCUE_COST_WEIGHT_DIST = 0.6
RESCUE_COST_WEIGHT_FEAT = 0.4
HEAVY_RESCUE_RADIUS = 25.0
HEAVY_LENGTH_TOL = 1.00
MUTUAL_NEAREST_IN_RESCUE = True

DMG_FEATURES = ["length", "radius_avg", "CSA_mm2", "volume_mm3", "surface_area", "tortuosity"]
DMG_MODE = "decrease_only"  # keep as-is

#MATCH_FRAC_THRESHOLD = 0.75  

THRESHOLDS_DEFAULT = [0.50, 0.75, 0.90]


#helpers functions

def _ensure_cols(df, cols):
    df = df.copy()
    for c, val in cols.items():
        if c not in df.columns:
            df[c] = val
    return df

def _first_attr(d, names, default=None, cast=float):
    for n in names:
        if n in d and d[n] not in (None, "", "NaN"):
            try:
                return cast(d[n])
            except Exception:
                try:
                    return float(d[n])
                except Exception:
                    return d[n]
    return default

def xyz(G, n):
    d = G.nodes[str(n)]
    x = _first_attr(d, ["x","X","coord_x","pos_x"], None)
    y = _first_attr(d, ["y","Y","coord_y","pos_y"], None)
    z = _first_attr(d, ["z","Z","coord_z","pos_z"], None)
    if None in (x,y,z):
        raise ValueError(f"Missing xyz for node: {n}")
    return np.array([float(x), float(y), float(z)], float)

def node_lobe(G, n):
    d = G.nodes[str(n)]
    lab = _first_attr(d, ["lobe","Lobe","label_lobe","lobe_label","region_lobe"], None, cast=str)
    if lab is None:
        return ""
    s = str(lab)
    return "Central" if s in CENTRAL_LABELS or s.lower() == "central" else s

def edge_feat(G, u, v, feat):
    d = G[str(u)][str(v)]
    if feat == "CSA_mm2":
        r = edge_feat(G, u, v, "radius_avg")
        if r is None or not np.isfinite(r):
            return np.nan
        return float(math.pi) * float(r) * float(r)

    alt = {
        "length": ["length","edge_length","len_mm","Length_mm","L"],
        "radius_avg": ["radius_avg","avg_radius","radius_mean","r_mean"],
        "surface_area": ["surface_area","surf_area","SA_mm2","surface_mm2"],
        "volume_mm3": ["volume_mm3","vol_mm3","volume","V_mm3"],
        "tortuosity": ["tortuosity","tau","Tortuosity"]
    }
    names = alt.get(feat, [feat])
    return _first_attr(d, names, default=np.nan)

def _edge_label_sanitized(G, u, v):
    d = G[str(u)][str(v)]
    if "lobe" in d:
        try:
            lv = float(d["lobe"])
            return "Central" if int(lv) == 0 else str(int(lv))
        except Exception:
            pass

    lab = d.get("label", None)
    if lab is None or lab == "":
        Lu, Lv = node_lobe(G, u), node_lobe(G, v)
        return str(Lu) if (Lu == Lv and Lu != "") else None

    s = str(lab)
    if s in {"0","central","Central"}:
        return "Central"
    if s == "dropped_from_central":
        return None
    return s

def _centroids(G, edges):
    if not edges:
        return np.zeros((0,3))
    return np.asarray([(xyz(G,u)+xyz(G,v))/2.0 for (u,v) in edges], float)

def _edge_tuples_sorted(edges):
    return {tuple(sorted((str(e[0]),str(e[1])))) for e in edges}

def _canon_pair_df(df, ucol, vcol):
    if df is None or df.empty:
        return df
    df = df.copy()
    a = df[ucol].astype(str)
    b = df[vcol].astype(str)
    umin = a.where(a <= b, b)
    vmax = b.where(a <= b, a)
    df[ucol] = umin
    df[vcol] = vmax
    return df

def _agg_damage_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["bl_u","bl_v","damaged","damage_features"])
    df = _canon_pair_df(df, "bl_u", "bl_v")
    return df.groupby(["bl_u","bl_v"], as_index=False).agg({
        "damaged": "max",
        "damage_features": lambda s: ",".join(sorted(set([t for x in s.fillna("") for t in x.split(",") if t])))
    })

def _umeyama_similarity(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY
    C = (Y0.T @ X0) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    varX = (X0 * X0).sum() / X.shape[0]
    s = np.trace(np.diag(S) @ D) / (varX + 1e-12)
    t = muY - s * (R @ muX)
    return float(s), R.astype(np.float64), t.astype(np.float64)

def _quick_build_anchors(G_BL, G_FU, max_n=PASS0_RANDOM_SUBSAMPLE):
    eBL = list(G_BL.edges())
    eFU = list(G_FU.edges())
    if len(eBL) < PASS0_MIN_ANCHORS or len(eFU) < PASS0_MIN_ANCHORS:
        return None, None
    Cbl = _centroids(G_BL, eBL)
    Cfu = _centroids(G_FU, eFU)
    if len(Cbl) > max_n:
        Cbl = Cbl[np.random.choice(len(Cbl), size=max_n, replace=False)]
    if len(Cfu) > max_n:
        Cfu = Cfu[np.random.choice(len(Cfu), size=max_n, replace=False)]
    tBL = cKDTree(Cbl)
    _, nn_idx = tBL.query(Cfu, k=1)
    if isinstance(nn_idx, np.ndarray) and nn_idx.shape[0] >= PASS0_MIN_ANCHORS:
        return Cfu, Cbl[nn_idx]
    return None, None

def _lsa_feature_cost(G_bl, G_fu, bl_pair, fu_pair):
    (bu, bv) = bl_pair
    (fu, fv) = fu_pair

    def _rel(a, b):
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
            return np.nan
        return abs(b - a) / (abs(a) + 1e-6)

    Lb = edge_feat(G_bl, bu, bv, "length")
    Lf = edge_feat(G_fu, fu, fv, "length")
    Rb = edge_feat(G_bl, bu, bv, "radius_avg")
    Rf = edge_feat(G_fu, fu, fv, "radius_avg")

    vals = []
    for v in (_rel(Lb, Lf), _rel(Rb, Rf)):
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0

def _estimate_gate_quick(C_fu, C_bl):
    if C_fu.size == 0 or C_bl.size == 0:
        return 12.0
    D = np.linalg.norm(C_fu[:, None, :] - C_bl[None, :, :], axis=2)
    q75 = np.nanpercentile(D, 75) if np.isfinite(D).any() else 12.0
    return float(max(8.0, min(24.0, 1.5 * q75)))

def _fast_lsa_by_label_featured(C_fu, C_bl, pairs_fu, pairs_bl, label, G_BL_local, G_FU_local):
    if C_fu.shape[0] == 0 or C_bl.shape[0] == 0:
        return [], [], {"lobe": label, "gate_mm": None, "FU_edges": C_fu.shape[0], "BL_edges": C_bl.shape[0], "matched": 0}, pd.DataFrame()

    gate = _estimate_gate_quick(C_fu, C_bl)
    tree_bl = cKDTree(C_bl)
    cand_map = {}
    bl_used = set()

    for i in range(C_fu.shape[0]):
        hits = tree_bl.query_ball_point(C_fu[i], r=gate)
        if not hits:
            continue
        if len(hits) > FAST_K_NEIGHBORS:
            dists = np.linalg.norm(C_bl[hits] - C_fu[i], axis=1)
            hits = [hits[j] for j in np.argsort(dists)[:FAST_K_NEIGHBORS]]
        cand_map[i] = hits
        bl_used.update(hits)

    if not cand_map:
        return [], list(range(C_fu.shape[0])), {"lobe": label, "gate_mm": gate, "FU_edges": C_fu.shape[0], "BL_edges": C_bl.shape[0], "matched": 0}, pd.DataFrame()

    bl_list = sorted(bl_used)
    bl_pos = {b: k for k, b in enumerate(bl_list)}
    C = np.full((C_fu.shape[0], len(bl_list)), 1e9, dtype=FAST_DTYPE)

    for i, hits in cand_map.items():
        dists = np.linalg.norm(C_bl[hits] - C_fu[i], axis=1).astype(np.float64)
        for hh, d in zip(hits, dists):
            total = d
            if USE_FEATURED_LSA and FEAT_COST_WEIGHT > 0:
                total = (1.0 - FEAT_COST_WEIGHT) * (d / (gate + 1e-6)) + FEAT_COST_WEIGHT * _lsa_feature_cost(
                    G_BL_local, G_FU_local, pairs_bl[hh], pairs_fu[i]
                )
            C[i, bl_pos[hh]] = FAST_DTYPE(total)

    r, c = linear_sum_assignment(C)

    rows = []
    fu_matched = set()
    prov = []

    for i, j in zip(r, c):
        raw_dist = float(np.linalg.norm(C_bl[bl_list[j]] - C_fu[i]))
        if raw_dist <= gate and C[i, j] < 1e8:
            fu_u, fu_v = pairs_fu[i]
            bl_u, bl_v = pairs_bl[bl_list[j]]
            rows.append({
                "lobe": label,
                "fu_u": str(fu_u), "fu_v": str(fu_v),
                "bl_u": str(bl_u), "bl_v": str(bl_v),
                "centroid_dist_mm": raw_dist,
                "gate_used_mm": float(gate)
            })
            prov.append({
                "lobe": label,
                "fu_u": str(fu_u), "fu_v": str(fu_v),
                "bl_u": str(bl_u), "bl_v": str(bl_v),
                "match_confidence": float(np.clip(1.0 - (raw_dist / (gate + 1e-6)), 0, 1)),
                "centroid_dist_mm": raw_dist,
                "length_diff_frac": np.nan,
                "radius_diff_frac": np.nan,
                "match_pass": "LSA",
                "is_mutual_nn": 0,
                "gate_used_mm": float(gate)
            })
            fu_matched.add(i)

    fu_unmatched = [ii for ii in range(C_fu.shape[0]) if ii not in fu_matched]
    gate_row = {"lobe": label, "gate_mm": float(gate), "FU_edges": int(C_fu.shape[0]), "BL_edges": int(C_bl.shape[0]), "matched": int(len(rows))}
    return rows, fu_unmatched, gate_row, pd.DataFrame(prov)

def rescue_with_mutual_nn(bl_leftover_set, fu_leftover_set, bl_xyz, fu_xyz, G_BL_local, G_FU_local,
                          radius_mm, length_tol_frac, mutual=True):
    if not bl_leftover_set or not fu_leftover_set:
        return [], [{"lobe": _edge_label_sanitized(G_BL_local, u, v), "bl_u": u, "bl_v": v} for (u, v) in bl_leftover_set], pd.DataFrame()

    fu_list = list(fu_leftover_set)
    fu_centroids = np.vstack([(fu_xyz[u] + fu_xyz[v]) * FAST_DTYPE(0.5) for (u, v) in fu_list])
    fu_lengths = np.asarray([edge_feat(G_FU_local, u, v, "length") for (u, v) in fu_list], dtype=FAST_DTYPE)
    fu_tree = cKDTree(fu_centroids)

    bl_list = list(bl_leftover_set)
    bl_centroids = np.vstack([(bl_xyz[u] + bl_xyz[v]) * FAST_DTYPE(0.5) for (u, v) in bl_list])
    bl_lengths = np.asarray([edge_feat(G_BL_local, u, v, "length") for (u, v) in bl_list], dtype=FAST_DTYPE)
    bl_tree = cKDTree(bl_centroids)

    fu_used = np.zeros(len(fu_centroids), dtype=bool)
    rescued = []
    disappeared = []
    prov = []

    for i, (u, v) in enumerate(bl_list):
        bl_c = bl_centroids[i]
        bl_L = float(bl_lengths[i]) if np.isfinite(bl_lengths[i]) else 0.0

        cand = fu_tree.query_ball_point(bl_c, r=radius_mm)
        cand = [j for j in cand if not fu_used[j]]
        if not cand:
            disappeared.append({"lobe": _edge_label_sanitized(G_BL_local, u, v), "bl_u": u, "bl_v": v})
            continue

        dists = np.linalg.norm(fu_centroids[cand] - bl_c, axis=1)
        rel = np.abs(fu_lengths[cand] - bl_L) / (abs(bl_L) + 1e-6) if bl_L != 0 else np.full(len(cand), np.inf)
        cost = RESCUE_COST_WEIGHT_DIST * (dists / (radius_mm + 1e-6)) + RESCUE_COST_WEIGHT_FEAT * rel
        jloc = int(np.argmin(cost))
        j = cand[jloc]

        if dists[jloc] < radius_mm and rel[jloc] < length_tol_frac:
            if mutual:
                idx_back = bl_tree.query(fu_centroids[j], k=1)[1]
                if idx_back != i:
                    disappeared.append({"lobe": _edge_label_sanitized(G_BL_local, u, v), "bl_u": u, "bl_v": v})
                    continue

            fu_used[j] = True
            fu_u, fu_v = fu_list[j]
            rescued.append({
                "lobe": _edge_label_sanitized(G_BL_local, u, v),
                "bl_u": u, "bl_v": v,
                "fu_u": fu_u, "fu_v": fu_v,
                "centroid_dist_mm": float(dists[jloc]),
                "gate_used_mm": float(radius_mm)
            })
            prov.append({
                "lobe": _edge_label_sanitized(G_BL_local, u, v),
                "bl_u": u, "bl_v": v,
                "fu_u": fu_u, "fu_v": fu_v,
                "match_confidence": float(np.clip(1.0 - (0.6 * (dists[jloc] / (radius_mm + 1e-6)) + 0.4 * rel[jloc]), 0.0, 1.0)),
                "centroid_dist_mm": float(dists[jloc]),
                "length_diff_frac": float(rel[jloc]),
                "radius_diff_frac": np.nan,
                "match_pass": "rescue_mutual",
                "is_mutual_nn": 1 if mutual else 0,
                "gate_used_mm": float(radius_mm)
            })
        else:
            disappeared.append({"lobe": _edge_label_sanitized(G_BL_local, u, v), "bl_u": u, "bl_v": v})

    return rescued, disappeared, pd.DataFrame(prov)

def compute_damage_flags(G_BL, G_FU, survived_df, drop_frac):
    rows = []
    if survived_df is None or survived_df.empty:
        return pd.DataFrame(columns=["lobe","bl_u","bl_v","fu_u","fu_v","damaged","damage_features"])

    for r in survived_df.itertuples(index=False):
        triggers = []
        for f in DMG_FEATURES:
            bl = edge_feat(G_BL, r.bl_u, r.bl_v, f)
            fu = edge_feat(G_FU, r.fu_u, r.fu_v, f)
            if not (np.isfinite(bl) and np.isfinite(fu)) or bl == 0:
                continue
            rel = (fu - bl) / bl
            if (DMG_MODE == "decrease_only" and rel <= -drop_frac) or (DMG_MODE != "decrease_only" and abs(rel) >= drop_frac):
                triggers.append(f)

        rows.append({
            "lobe": r.lobe,
            "bl_u": r.bl_u, "bl_v": r.bl_v,
            "fu_u": r.fu_u, "fu_v": r.fu_v,
            "damaged": bool(len(triggers) > 0),
            "damage_features": ",".join(sorted(set(triggers)))
        })

    return pd.DataFrame(rows)

def per_lobe_counts(survived, disappeared, damaged_df):
    survived = _ensure_cols(survived, {"lobe": ""})
    disappeared = _ensure_cols(disappeared, {"lobe": ""})

    s = survived.groupby("lobe").size().rename("survived") if not survived.empty else pd.Series(dtype=int, name="survived")
    d = disappeared.groupby("lobe").size().rename("disappeared") if not disappeared.empty else pd.Series(dtype=int, name="disappeared")
    out = pd.concat([s, d], axis=1).fillna(0).astype(int).reset_index().rename(columns={"index": "lobe"})

    if damaged_df is not None and not damaged_df.empty:
        k = damaged_df[damaged_df["damaged"] == True]
        if not k.empty:
            dam_cnt = k.groupby("lobe").size().rename("damaged")

            def _join_feats(ss):
                toks = []
                for s_ in ss.fillna(""):
                    if not s_:
                        continue
                    toks.extend([t.strip() for t in s_.split(",") if t.strip()])
                return ",".join(sorted(set(toks)))

            feat_lists = k.groupby("lobe")["damage_features"].apply(_join_feats).rename("damaged_features_used")
            out = out.merge(dam_cnt, on="lobe", how="left").merge(feat_lists, on="lobe", how="left")
        else:
            out["damaged"] = 0
            out["damaged_features_used"] = ""
    else:
        out["damaged"] = 0
        out["damaged_features_used"] = ""

    for c in ["survived", "disappeared", "damaged"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype(int)

    if "damaged_features_used" not in out.columns:
        out["damaged_features_used"] = ""
    return out.sort_values("lobe").reset_index(drop=True)

def write_labeled_graphs_single_label(G_BL, G_FU, survived, disappeared, damaged_df, pair_dir: Path):
    pair_dir.mkdir(parents=True, exist_ok=True)

    H_BL = G_BL.copy()
    H_FU = G_FU.copy()
    for u, v in H_BL.edges():
        H_BL[u][v]["final_label"] = ""
    for u, v in H_FU.edges():
        H_FU[u][v]["final_label"] = ""

    survived_c = _canon_pair_df(survived, "bl_u", "bl_v") if survived is not None else pd.DataFrame()
    disappeared_c = _canon_pair_df(disappeared, "bl_u", "bl_v") if disappeared is not None else pd.DataFrame()
    damaged_c = _agg_damage_df(damaged_df) if (damaged_df is not None and not damaged_df.empty) else pd.DataFrame(columns=["bl_u","bl_v","damaged","damage_features"])

    disappeared_set = set(map(tuple, disappeared_c[["bl_u","bl_v"]].astype(str).values)) if not disappeared_c.empty else set()
    damaged_set = set(map(tuple, damaged_c[damaged_c["damaged"] == True][["bl_u","bl_v"]].astype(str).values)) if not damaged_c.empty else set()
    survived_set = set(map(tuple, survived_c[["bl_u","bl_v"]].astype(str).values)) if not survived_c.empty else set()

    bl_all = _edge_tuples_sorted(H_BL.edges())
    for (u, v) in bl_all:
        label = "survived"
        if (u, v) in disappeared_set:
            label = "disappeared"
        elif (u, v) in damaged_set:
            label = "damaged"
        elif (u, v) in survived_set:
            label = "survived"
        else:
            label = "disappeared"

        if H_BL.has_edge(u, v):
            H_BL[u][v]["final_label"] = label
        if H_BL.has_edge(v, u):
            H_BL[v][u]["final_label"] = label

    if survived is not None and not survived.empty:
        for r in survived.itertuples(index=False):
            u, v = str(r.fu_u), str(r.fu_v)
            if H_FU.has_edge(u, v):
                H_FU[u][v]["final_label"] = "survived"
            if H_FU.has_edge(v, u):
                H_FU[v][u]["final_label"] = "survived"

    nx.write_graphml(H_BL, pair_dir / "BL_labeled.graphml")
    nx.write_graphml(H_FU, pair_dir / "FU_labeled.graphml")

def _model_rows_for_pair(G_BL, pair_dir: Path, survived_df, disappeared_df, damage_df, agg_csv_path: Path):
    survived_df = _canon_pair_df(_ensure_cols(survived_df, {"bl_u":"", "bl_v":"", "fu_u":"", "fu_v":""}), "bl_u", "bl_v") if survived_df is not None else pd.DataFrame()
    disappeared_df = _canon_pair_df(_ensure_cols(disappeared_df, {"bl_u":"", "bl_v":""}), "bl_u", "bl_v") if disappeared_df is not None else pd.DataFrame()
    damage_df = _agg_damage_df(_ensure_cols(damage_df, {"bl_u":"", "bl_v":"", "damaged":False, "damage_features":""})) if damage_df is not None else pd.DataFrame()

    bl_all = _edge_tuples_sorted(G_BL.edges())
    bl_all_rows = pd.DataFrame([{"bl_u": u, "bl_v": v} for (u, v) in bl_all])

    surv_mark = survived_df.assign(_survived=1)[["bl_u","bl_v","fu_u","fu_v","_survived"]] if not survived_df.empty else pd.DataFrame(columns=["bl_u","bl_v","fu_u","fu_v","_survived"])
    bl_all_rows = bl_all_rows.merge(surv_mark, how="left", on=["bl_u","bl_v"])

    disc_mark = disappeared_df.assign(_disappeared=1)[["bl_u","bl_v","_disappeared"]] if not disappeared_df.empty else pd.DataFrame(columns=["bl_u","bl_v","_disappeared"])
    bl_all_rows = bl_all_rows.merge(disc_mark, how="left", on=["bl_u","bl_v"])

    bl_all_rows["_survived"] = bl_all_rows["_survived"].fillna(0).astype(int)
    bl_all_rows["_disappeared"] = bl_all_rows["_disappeared"].fillna(0).astype(int)

    if not damage_df.empty:
        bl_all_rows = bl_all_rows.merge(damage_df[["bl_u","bl_v","damaged","damage_features"]], how="left", on=["bl_u","bl_v"])
    else:
        bl_all_rows["damaged"] = False
        bl_all_rows["damage_features"] = ""

    bl_all_rows.loc[bl_all_rows["_survived"] != 1, ["damaged", "damage_features"]] = [False, ""]

    def _final_label_bl(r):
        if r["_disappeared"] == 1:
            return "disappeared"
        if r["_survived"] == 1 and bool(r["damaged"]):
            return "damaged"
        if r["_survived"] == 1:
            return "survived"
        return "disappeared"

    bl_all_rows["final_label_bl"] = bl_all_rows.apply(_final_label_bl, axis=1)

    bl_all_rows["label_damaged"] = bl_all_rows["final_label_bl"].isin(["disappeared","damaged"]).astype(int)
    bl_all_rows["label_survived"] = 1 - bl_all_rows["label_damaged"]

    parts = pair_dir.parts
    patient, modality, timepoint = parts[-3], parts[-2], parts[-1]
    bl_all_rows.insert(0, "patient", patient)
    bl_all_rows.insert(1, "modality", modality)
    bl_all_rows.insert(2, "timepoint", timepoint)

    bl_all_rows["side"] = [
        G_BL[str(u)][str(v)].get("side", "") if G_BL.has_edge(str(u), str(v)) else ""
        for u, v in bl_all_rows[["bl_u","bl_v"]].values
    ]

    out_csv = pair_dir / "edges_bl_fu_modeling_rows.csv"
    bl_all_rows.to_csv(out_csv, index=False)

    agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not agg_csv_path.exists()
    bl_all_rows.to_csv(agg_csv_path, mode=("w" if header_needed else "a"), header=header_needed, index=False)

def _plot_overlay(G_BL, G_FU, out_png: Path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Labeled changes (BL frame)")

    def _seg(ax_, p, q, lw=1.0, a=1.0, color="k"):
        ax_.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], lw=lw, alpha=a, color=color)

    for u, v, d in G_BL.edges(data=True):
        lab = str(d.get("final_label", "")).strip()
        if lab == "survived":
            _seg(ax, xyz(G_BL, u), xyz(G_BL, v), lw=1.0, a=0.90, color="#1f77b4")
        elif lab == "disappeared":
            _seg(ax, xyz(G_BL, u), xyz(G_BL, v), lw=1.6, a=0.85, color="#d62728")
        elif lab == "damaged":
            _seg(ax, xyz(G_BL, u), xyz(G_BL, v), lw=2.2, a=0.95, color="#7f3c8d")

    ax.legend(handles=[
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="Survived"),
        Line2D([0], [0], color="#d62728", lw=2.0, label="Disappeared"),
        Line2D([0], [0], color="#7f3c8d", lw=3.0, label="Damaged")
    ], loc="upper left", bbox_to_anchor=(0, 1.02))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _pair_report_row(pair_dir: Path, drop_frac: float, survived_df: pd.DataFrame, disappeared_df: pd.DataFrame,
                     dmg_df: pd.DataFrame, prov_df: pd.DataFrame, n_bl_edges: int, n_fu_edges: int):
    parts = pair_dir.parts
    patient, modality, timepoint = parts[-3], parts[-2], parts[-1]

    if prov_df is None or prov_df.empty or "match_confidence" not in prov_df.columns:
        match_frac = 0.0
        mean_conf = np.nan
        med_conf = np.nan
        mean_dist = np.nan
        med_dist = np.nan
        mean_gate = np.nan
    else:
        mc = pd.to_numeric(prov_df["match_confidence"], errors="coerce").dropna()
        dd = pd.to_numeric(prov_df.get("centroid_dist_mm", np.nan), errors="coerce")
        gg = pd.to_numeric(prov_df.get("gate_used_mm", np.nan), errors="coerce")
        match_frac = float(mc.mean()) if len(mc) else 0.0
        mean_conf = float(mc.mean()) if len(mc) else np.nan
        med_conf = float(mc.median()) if len(mc) else np.nan
        mean_dist = float(dd.mean()) if np.isfinite(dd).any() else np.nan
        med_dist = float(dd.median()) if np.isfinite(dd).any() else np.nan
        mean_gate = float(gg.mean()) if np.isfinite(gg).any() else np.nan

    n_surv = int(len(survived_df)) if survived_df is not None else 0
    n_dis = int(len(disappeared_df)) if disappeared_df is not None else 0
    n_dmg = int((dmg_df["damaged"] == True).sum()) if (dmg_df is not None and "damaged" in dmg_df.columns) else 0

    return {
        "patient": patient,
        "modality": modality,
        "timepoint": timepoint,
        "drop_threshold": float(drop_frac),
        "match_frac": float(match_frac),
        "mean_match_confidence": mean_conf,
        "median_match_confidence": med_conf,
        "mean_centroid_dist_mm": mean_dist,
        "median_centroid_dist_mm": med_dist,
        "mean_gate_used_mm": mean_gate,
        "matched_edges": n_surv,
        "disappeared_edges": n_dis,
        "damaged_edges": n_dmg,
        "bl_edges_total": int(n_bl_edges),
        "fu_edges_total": int(n_fu_edges),
    }

def _append_report_row(report_csv: Path, row: dict):
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header_needed = not report_csv.exists()
    df.to_csv(report_csv, mode=("w" if header_needed else "a"), header=header_needed, index=False)

#pair runner 

def run_pair(BL_path: Path, FU_path: Path, save_pair_dir: Path, agg_csv_path: Path,
             report_csv_path: Path, drop_frac: float, show_inline: bool = False):
    save_pair_dir.mkdir(parents=True, exist_ok=True)

    G_BL = nx.read_graphml(BL_path)
    G_FU = nx.read_graphml(FU_path)

    n_bl_edges = int(G_BL.number_of_edges())
    n_fu_edges = int(G_FU.number_of_edges())

    bl_xyz = {str(n): xyz(G_BL, n).astype(FAST_DTYPE) for n in G_BL.nodes()}
    fu_xyz = {str(n): xyz(G_FU, n).astype(FAST_DTYPE) for n in G_FU.nodes()}

    if USE_PASS0_SIMILARITY:
        Cfu, Cbl_nn = _quick_build_anchors(G_BL, G_FU)
        if Cfu is not None and Cbl_nn is not None:
            s, R, t = _umeyama_similarity(Cfu, Cbl_nn)
            Rt = R.T
            for n, p in fu_xyz.items():
                fu_xyz[n] = (s * (p @ Rt) + t).astype(FAST_DTYPE)

    def _build_side(G, xyz_map):
        pairs = [(str(u), str(v)) for (u, v) in G.edges()]
        cents = np.vstack([(xyz_map[u] + xyz_map[v]) * FAST_DTYPE(0.5) for (u, v) in pairs]) if len(pairs) else np.zeros((0, 3), dtype=FAST_DTYPE)
        by_label = {}
        for i, (u, v) in enumerate(pairs):
            lab = _edge_label_sanitized(G, u, v)
            if lab is None:
                continue
            by_label.setdefault(str(lab), []).append(i)
        return {"pairs": pairs, "centroids": cents, "idx_by_label": by_label}

    BL = _build_side(G_BL, bl_xyz)
    FU = _build_side(G_FU, fu_xyz)

    if not BL["pairs"] or not FU["pairs"]:
        survived_df = pd.DataFrame(columns=["lobe","bl_u","bl_v","fu_u","fu_v"])
        disappeared_df = pd.DataFrame([{"lobe": _edge_label_sanitized(G_BL, u, v), "bl_u": u, "bl_v": v}
                                       for (u, v) in _edge_tuples_sorted(BL["pairs"])])

        survived_df.to_csv(save_pair_dir / "survived_edges.csv", index=False)
        disappeared_df.to_csv(save_pair_dir / "disappeared_edges.csv", index=False)

        dmg_df = pd.DataFrame(columns=["lobe","bl_u","bl_v","fu_u","fu_v","damaged","damage_features"])
        dmg_df.to_csv(save_pair_dir / "matched_edges_damage_flags.csv", index=False)

        per_lobe = per_lobe_counts(survived_df, disappeared_df, dmg_df)
        per_lobe.to_csv(save_pair_dir / "change_counts_by_lobe.csv", index=False)

        write_labeled_graphs_single_label(G_BL, G_FU, survived_df, disappeared_df, dmg_df, save_pair_dir)
        _model_rows_for_pair(G_BL, save_pair_dir, survived_df, disappeared_df, dmg_df, agg_csv_path)

        prov_df = pd.DataFrame(columns=["match_confidence","centroid_dist_mm","gate_used_mm"])
        prov_df.to_csv(save_pair_dir / "match_provenance.csv", index=False)

        rep_row = _pair_report_row(save_pair_dir, drop_frac, survived_df, disappeared_df, dmg_df, prov_df, n_bl_edges, n_fu_edges)
        _append_report_row(report_csv_path, rep_row)

        try:
            _plot_overlay(nx.read_graphml(save_pair_dir / "BL_labeled.graphml"),
                          nx.read_graphml(save_pair_dir / "FU_labeled.graphml"),
                          save_pair_dir / "overlay_labeled_changes.png")
        except Exception:
            pass
        return

    survived_rows = []
    prov_all = []  # FIX: was referenced later but not defined in your pasted code

#central
    prov_rows = []
    idx_bl = BL["idx_by_label"].get("Central", [])
    idx_fu = FU["idx_by_label"].get("Central", [])
    pairs_bl = [BL["pairs"][i] for i in idx_bl]
    pairs_fu = [FU["pairs"][i] for i in idx_fu]
    C_bl = BL["centroids"][idx_bl] if len(idx_bl) else np.zeros((0,3), dtype=FAST_DTYPE)
    C_fu = FU["centroids"][idx_fu] if len(idx_fu) else np.zeros((0,3), dtype=FAST_DTYPE)
    rows_c, fu_un_c, gate_c, prov_c = _fast_lsa_by_label_featured(C_fu, C_bl, pairs_fu, pairs_bl, "Central", G_BL, G_FU)
    if rows_c:
        survived_rows.extend(rows_c)
    if not prov_c.empty:
        prov_rows.append(prov_c)

#lobes
    lobes = sorted({k for k in (set(BL["idx_by_label"].keys()) | set(FU["idx_by_label"].keys())) if k != "Central"})
    for L in lobes:
        idx_bl = BL["idx_by_label"].get(L, [])
        idx_fu = FU["idx_by_label"].get(L, [])
        pairs_bl = [BL["pairs"][i] for i in idx_bl]
        pairs_fu = [FU["pairs"][i] for i in idx_fu]
        C_bl = BL["centroids"][idx_bl] if len(idx_bl) else np.zeros((0,3), dtype=FAST_DTYPE)
        C_fu = FU["centroids"][idx_fu] if len(idx_fu) else np.zeros((0,3), dtype=FAST_DTYPE)
        rows_L, fu_un_L, gate_L, prov_L = _fast_lsa_by_label_featured(C_fu, C_bl, pairs_fu, pairs_bl, L, G_BL, G_FU)
        if rows_L:
            survived_rows.extend(rows_L)
        if not prov_L.empty:
            prov_rows.append(prov_L)

    survived_df = pd.DataFrame(survived_rows)

    bl_survived_set = _edge_tuples_sorted(survived_df[["bl_u","bl_v"]].values) if not survived_df.empty else set()
    bl_leftover_set = _edge_tuples_sorted(BL["pairs"]) - bl_survived_set

    fu_survived_set = _edge_tuples_sorted(survived_df[["fu_u","fu_v"]].values) if not survived_df.empty else set()
    fu_leftover_set = _edge_tuples_sorted(FU["pairs"]) - fu_survived_set

    rescued_rows, disappeared_rows, prov_res = rescue_with_mutual_nn(
        bl_leftover_set, fu_leftover_set,
        bl_xyz={str(n): xyz(G_BL, n).astype(FAST_DTYPE) for n in G_BL.nodes()},
        fu_xyz={str(n): xyz(G_FU, n).astype(FAST_DTYPE) for n in G_FU.nodes()},
        G_BL_local=G_BL, G_FU_local=G_FU,
        radius_mm=HEAVY_RESCUE_RADIUS, length_tol_frac=HEAVY_LENGTH_TOL,
        mutual=MUTUAL_NEAREST_IN_RESCUE
    )

    if rescued_rows:
        survived_df = pd.concat([survived_df, pd.DataFrame(rescued_rows)], ignore_index=True)

    disappeared_df = pd.DataFrame(disappeared_rows)[["lobe","bl_u","bl_v"]] if disappeared_rows else pd.DataFrame(columns=["lobe","bl_u","bl_v"])

    survived_df = _ensure_cols(_canon_pair_df(survived_df, "bl_u","bl_v"), {"lobe":"", "bl_u":"", "bl_v":"", "fu_u":"", "fu_v":""})
    disappeared_df = _ensure_cols(_canon_pair_df(disappeared_df, "bl_u","bl_v"), {"lobe":"", "bl_u":"", "bl_v":""})

    survived_df.to_csv(save_pair_dir / "survived_edges.csv", index=False)
    disappeared_df.to_csv(save_pair_dir / "disappeared_edges.csv", index=False)

    dmg_df = compute_damage_flags(G_BL, G_FU, survived_df, drop_frac=drop_frac)
    dmg_df = _ensure_cols(dmg_df, {"lobe":"", "bl_u":"", "bl_v":"", "fu_u":"", "fu_v":"", "damaged":False, "damage_features":""})
    dmg_df.to_csv(save_pair_dir / "matched_edges_damage_flags.csv", index=False)

    per_lobe = per_lobe_counts(survived_df, disappeared_df, dmg_df)
    per_lobe.to_csv(save_pair_dir / "change_counts_by_lobe.csv", index=False)

    write_labeled_graphs_single_label(G_BL, G_FU, survived_df, disappeared_df, dmg_df, save_pair_dir)
    _model_rows_for_pair(G_BL, save_pair_dir, survived_df, disappeared_df, dmg_df, agg_csv_path)

#quality report
    if prov_rows:
        prov_all.append(pd.concat(prov_rows, ignore_index=True))
    if prov_res is not None and not prov_res.empty:
        prov_all.append(prov_res)

    prov_df = pd.concat(prov_all, ignore_index=True) if prov_all else pd.DataFrame()
    prov_df.to_csv(save_pair_dir / "match_provenance.csv", index=False)

    rep_row = _pair_report_row(
        save_pair_dir, drop_frac,
        survived_df=survived_df,
        disappeared_df=disappeared_df,
        dmg_df=dmg_df,
        prov_df=prov_df,
        n_bl_edges=n_bl_edges,
        n_fu_edges=n_fu_edges
    )
    _append_report_row(report_csv_path, rep_row)

    try:
        G_BL_lab = nx.read_graphml(save_pair_dir / "BL_labeled.graphml")
        G_FU_lab = nx.read_graphml(save_pair_dir / "FU_labeled.graphml")
        _plot_overlay(G_BL_lab, G_FU_lab, save_pair_dir / "overlay_labeled_changes.png")
    except Exception as e:
        print(f"  [Warn] Overlay failed: {e}")



#pair discovery runners

def discover_pairs(input_root: Path, valid_modalities):
    pairs = []
    for pat_dir in sorted([p for p in input_root.glob("P*/") if p.is_dir()]):
        for mod_dir in sorted([m for m in pat_dir.iterdir() if m.is_dir() and m.name in set(valid_modalities)]):
            bl = mod_dir / "BASELINE" / "sanitize" / "central_sanitized.graphml"
            if not bl.exists():
                continue
            for tp_dir in sorted([t for t in mod_dir.iterdir() if t.is_dir() and t.name.upper().startswith("FU")]):
                fu = tp_dir / "prematch_rigid" / "FU_sanitized_shifted.graphml"
                if fu.exists():
                    pairs.append((pat_dir.name, mod_dir.name, tp_dir.name, bl, fu))
    return pairs

def run_all_thresholds(
    input_root: Path,
    save_root_base: Path,
    thresholds,
    valid_modalities=None,
    show_inline_first_n: int = SHOW_INLINE_FIRST_N_DEFAULT
):
    valid_modalities = valid_modalities or VALID_MODALITIES_DEFAULT

    pairs = discover_pairs(input_root, valid_modalities)
    print(f"Discovered {len(pairs)} pair(s) to match\n")

    for drop_frac in thresholds:
        tag = f"damage_{int(round(drop_frac * 100))}pct"
        save_root = save_root_base / tag
        save_root.mkdir(parents=True, exist_ok=True)

        agg_csv = save_root / "_edges_bl_fu_modeling_aggregate.csv"

        report_dir = save_root / "_postmatch_reports"
        report_csv = report_dir / "lobe_counts_and_quality.csv"
        if report_csv.exists():
            report_csv.unlink()

        print(f"\nDAMAGE DROP THRESHOLD = {int(round(drop_frac * 100))}%")
        print(f"Report CSV (match quality): {report_csv}")

        shown = 0
        for k, (pid, mod, tp, bl, fu) in enumerate(pairs, 1):
            out_dir = save_root / pid / mod / tp
            print(f"[{k}/{len(pairs)}] {pid} | {mod} | {tp} — matching… (drop={drop_frac:.2f})")
            try:
                run_pair(
                    BL_path=bl,
                    FU_path=fu,
                    save_pair_dir=out_dir,
                    agg_csv_path=agg_csv,
                    report_csv_path=report_csv,
                    drop_frac=drop_frac,
                    show_inline=(shown < show_inline_first_n)
                )
                shown += 1

                counts_path = out_dir / "change_counts_by_lobe.csv"
                counts = pd.read_csv(counts_path) if counts_path.exists() else pd.DataFrame()
                s = int(counts["survived"].sum()) if "survived" in counts else 0
                d = int(counts["disappeared"].sum()) if "disappeared" in counts else 0
                g = int(counts["damaged"].sum()) if "damaged" in counts else 0

                mf = ""
                try:
                    rep = pd.read_csv(report_csv)
                    rep_last = rep.iloc[-1]
                    mf = f" | match_frac={float(rep_last['match_frac']):.3f}"
                except Exception:
                    mf = ""

                print(f"    survived={s}  disappeared={d}  damaged={g}{mf}")
            except Exception as e:
                print(f"!!! ERROR [{k}/{len(pairs)}] {pid} | {mod} | {tp}: {e}")

    print("\nAll thresholds finished")



#CLI

def _parse_args():
    p = argparse.ArgumentParser(description="Baseline↔Follow-up vessel graph matching + labeling (LSA + rescue + damage flags).")
    p.add_argument("--input-root", type=str, required=True, help="Root directory containing P*/(Artery|Vein)/... structure.")
    p.add_argument("--save-root", type=str, required=True, help="Base output directory; subfolders per threshold will be created.")
    p.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS_DEFAULT, help="Damage drop thresholds, e.g. 0.5 0.75 0.9")
    p.add_argument("--modalities", type=str, nargs="+", default=VALID_MODALITIES_DEFAULT, help="Modalities to process, e.g. Artery Vein")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed (affects PASS0 subsampling).")
    p.add_argument("--show-inline-first-n", type=int, default=SHOW_INLINE_FIRST_N_DEFAULT, help="Kept for compatibility; does not display inline in script mode.")
    return p.parse_args()

def main():
    args = _parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    run_all_thresholds(
        input_root=Path(args.input_root),
        save_root_base=Path(args.save_root),
        thresholds=args.thresholds,
        valid_modalities=args.modalities,
        show_inline_first_n=args.show_inline_first_n
    )

if __name__ == "__main__":
    main()

