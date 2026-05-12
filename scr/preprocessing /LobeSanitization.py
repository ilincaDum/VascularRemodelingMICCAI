"""

This script sanitizes "Central" (label=0) edges in GraphML vessel graphs by:
- selecting anchor central edges (largest radius_avg),
- building a central-only subgraph filtered by radius threshold,
- computing multi-source geodesic distances from anchor nodes,
- keeping central edges whose endpoints lie in the reachable "core",
- otherwise relabeling central edges to the nearest lobar cluster (same-side),
- writing:
    - central_sanitized.graphml
    - central_sanitize_audit.csv
    - central_sanitize_decisions.json

"""

import argparse
import ast
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LEFT_LOBES = {1, 2, "1", "2"}
RIGHT_LOBES = {3, 4, 5, "3", "4", "5"}
CENTRAL_LABELS = {0, "0", "Central", "central", ""}

CAND_COORDS = [("X", "Y", "Z"), ("x", "y", "z"), ("x_mm", "y_mm", "z_mm"), ("xc", "yc", "zc")]


def _infer_side_from_lobe(L):
    if L in LEFT_LOBES:
        return "Left"
    if L in RIGHT_LOBES:
        return "Right"
    return "Central"


def node_xyz(G, n):
    d = G.nodes[str(n)]
    for kx, ky, kz in CAND_COORDS:
        if kx in d and ky in d and kz in d:
            try:
                return float(d[kx]), float(d[ky]), float(d[kz])
            except Exception:
                pass
    for k in ("pos", "coord", "coords", "point", "centroid"):
        if k in d:
            val = d[k]
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                try:
                    return float(val[0]), float(val[1]), float(val[2])
                except Exception:
                    pass
            try:
                tup = ast.literal_eval(str(val))
                if isinstance(tup, (list, tuple)) and len(tup) >= 3:
                    return float(tup[0]), float(tup[1]), float(tup[2])
            except Exception:
                pass
    return None


def xyz(G, n):
    p = node_xyz(G, n)
    if p is None or any(not np.isfinite(v) for v in p):
        raise ValueError(f"Missing/invalid xyz for node {n}")
    return np.asarray(p, dtype=float)


def node_side(G, n):
    d = G.nodes[str(n)]
    s = d.get("side", None)
    if s in ("Left", "Right"):
        return s
    L = d.get("lobe", d.get("label", d.get("region", "")))
    return _infer_side_from_lobe(L)


def _edge_data(G, u, v):
    u = str(u)
    v = str(v)
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        ed = G.get_edge_data(u, v)
        if not ed:
            return {}
        k0 = next(iter(ed.keys()))
        return ed[k0] if isinstance(ed[k0], dict) else {}
    return G[u][v] if G.has_edge(u, v) else {}


def _edge_centroid_xyz(G, u, v):
    return (xyz(G, u) + xyz(G, v)) / 2.0


def _edge_attr_float(G, u, v, key, default=np.nan):
    try:
        val = _edge_data(G, u, v).get(key, default)
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _edge_length_mm(G, u, v):
    L = _edge_attr_float(G, u, v, "length", np.nan)
    if not np.isfinite(L):
        L = float(np.linalg.norm(xyz(G, u) - xyz(G, v)))
    return float(L)


def _edge_radius_avg(G, u, v):
    return _edge_attr_float(G, u, v, "radius_avg", np.nan)


def edges_by_lobe(G):
    out = defaultdict(list)
    for e in G.edges(data=True):
        if len(e) == 3:
            u, v, ed = e
        else:
            u, v, *_ = e
            ed = _edge_data(G, u, v)

        u = str(u)
        v = str(v)

        lab = ed.get("label", ed.get("lobe", None))
        if lab is None or str(lab).strip() == "":
            Lu = G.nodes[u].get("lobe", "")
            Lv = G.nodes[v].get("lobe", "")
            lab = Lu if (Lu == Lv) else ""

        if lab in CENTRAL_LABELS or str(lab).strip().lower() == "central":
            L = "Central"
        else:
            try:
                L = str(int(float(lab)))
            except Exception:
                L = str(lab)

        out[L].append((u, v))
    return dict(out)


def collect_mixed_side_edges(G, allow_central_as_neutral=True):
    bad = []
    for u, v in G.edges():
        u = str(u)
        v = str(v)
        su = node_side(G, u)
        sv = node_side(G, v)

        if allow_central_as_neutral and ("Central" in (su, sv)):
            continue

        if su in ("Left", "Right") and sv in ("Left", "Right") and su != sv:
            bad.append(
                (u, v, su, sv, G.nodes[u].get("lobe", ""), G.nodes[v].get("lobe", ""))
            )
    return bad


def normalize_graph_sides_from_lobes(G):
    changed = 0
    for n in G.nodes():
        n = str(n)
        L = G.nodes[n].get("lobe", "")
        inferred = _infer_side_from_lobe(L)
        if inferred in ("Left", "Right") and G.nodes[n].get("side", None) != inferred:
            G.nodes[n]["side"] = inferred
            changed += 1
    return changed


def _central_edges(G):
    out = []
    for L, es in edges_by_lobe(G).items():
        if L in ("Central",) or L in CENTRAL_LABELS:
            out.extend(es)
    return [tuple(sorted((str(u), str(v)))) for (u, v) in out]


def _build_lobe_kdtrees_same_graph(G):
    grp = edges_by_lobe(G)
    lobar = {}
    for L, es in grp.items():
        if L in ("Central",) or L in CENTRAL_LABELS or L == "":
            continue
        C = np.asarray([_edge_centroid_xyz(G, u, v) for (u, v) in es], float) if es else np.zeros((0, 3))
        lobar[L] = dict(edges=es, C=C, tree=(cKDTree(C) if len(C) else None))
    return lobar


def _edge_side_hint(G, u, v):
    u = str(u)
    v = str(v)
    su, sv = (G.nodes[u].get("side")), (G.nodes[v].get("side"))
    if su in ("Left", "Right") and su == sv:
        return su

    votes = []
    for n in (u, v):
        for nbr in G.neighbors(n):
            s = G.nodes[str(nbr)].get("side")
            if s in ("Left", "Right"):
                votes.append(s)
    if votes:
        return "Left" if votes.count("Left") >= votes.count("Right") else "Right"
    return None


def _same_side(G, u, v, target_lobe):
    lside = _infer_side_from_lobe(target_lobe)
    if not lside or lside == "Central":
        return True
    return (node_side(G, u) == lside) and (node_side(G, v) == lside)


def sanitize_central_by_anchor_connectivity(
    G,
    num_anchor_edges=5,
    radius_floor_factor=0.7,
    geodesic_limit_mm=40.0,
    far_gate_mm=None,
    allow_drop_if_no_lobe=True,
):
    cent_es = _central_edges(G)
    if not cent_es:
        return [], {}, [], [], []

    radii = [(_edge_radius_avg(G, u, v), (u, v)) for (u, v) in cent_es]
    finite = [(r, e) for (r, e) in radii if np.isfinite(r)]
    finite.sort(key=lambda x: x[0], reverse=True)
    anchors = [e for (_, e) in finite[: max(1, int(num_anchor_edges))]]

    med_r = np.median([r for (r, _) in finite]) if finite else 0.0
    r_floor = float(med_r) * float(radius_floor_factor)

    H = nx.Graph()
    for (u, v) in cent_es:
        r = _edge_radius_avg(G, u, v)
        if np.isfinite(r) and (r >= r_floor):
            H.add_edge(u, v, length=_edge_length_mm(G, u, v))

    anchor_nodes = set()
    for (u, v) in anchors:
        anchor_nodes.update([u, v])

    if not anchor_nodes or H.number_of_edges() == 0:
        dist_map = {}
        reachable = set()
    else:
        dist_map = nx.multi_source_dijkstra_path_length(H, anchor_nodes, weight="length")
        reachable = {n for (n, d) in dist_map.items() if d <= float(geodesic_limit_mm)}

    lobar_trees = _build_lobe_kdtrees_same_graph(G)
    kept, dropped = [], []
    relabeled = {L: [] for L in lobar_trees.keys()}
    audit = []

    def _nearest_lobe_local(u, v):
        c = _edge_centroid_xyz(G, u, v)
        best_L, best_d = None, np.inf
        side_hint = _edge_side_hint(G, u, v)

        for L, pack in lobar_trees.items():
            if pack["tree"] is None:
                continue
            Lside = _infer_side_from_lobe(L)
            if side_hint and Lside != side_hint:
                continue
            d, _ = pack["tree"].query(c, k=1)
            if not _same_side(G, u, v, L):
                d *= 1.35
            if d < best_d:
                best_d, best_L = d, L

        if best_L is not None:
            return best_L, float(best_d)

        for L, pack in lobar_trees.items():
            if pack["tree"] is None:
                continue
            d, _ = pack["tree"].query(c, k=1)
            d = float(d) * 10.0
            if d < best_d:
                best_d, best_L = d, L

        return best_L, float(best_d)

    for (u, v) in cent_es:
        core = (u in reachable) and (v in reachable)

        if core:
            kept.append((u, v))
            audit.append(
                {
                    "u": u,
                    "v": v,
                    "decision": "keep_central_via_anchor_core",
                    "radius_avg": _edge_radius_avg(G, u, v),
                    "geodesic_to_anchor_mm": max(dist_map.get(u, np.inf), dist_map.get(v, np.inf)),
                    "r_floor": r_floor,
                }
            )
            continue

        Lbest, d_l = _nearest_lobe_local(u, v)
        if Lbest is not None:
            if far_gate_mm is not None and np.isfinite(d_l) and (d_l > float(far_gate_mm)):
                if allow_drop_if_no_lobe:
                    dropped.append((u, v))
                    audit.append(
                        {"u": u, "v": v, "decision": "drop_from_central_far_gate", "nearest_lobe_mm": d_l}
                    )
                else:
                    kept.append((u, v))
                    audit.append(
                        {"u": u, "v": v, "decision": "keep_central_far_gate_fallback", "nearest_lobe_mm": d_l}
                    )
            else:
                relabeled[Lbest].append((u, v))
                audit.append(
                    {
                        "u": u,
                        "v": v,
                        "decision": "relabel_to_lobe",
                        "new_lobe": str(Lbest),
                        "radius_avg": _edge_radius_avg(G, u, v),
                        "nearest_lobe_mm": d_l,
                    }
                )
        else:
            if allow_drop_if_no_lobe:
                dropped.append((u, v))
                audit.append({"u": u, "v": v, "decision": "drop_from_central", "radius_avg": _edge_radius_avg(G, u, v)})
            else:
                kept.append((u, v))
                audit.append({"u": u, "v": v, "decision": "keep_central_fallback", "radius_avg": _edge_radius_avg(G, u, v)})

    return kept, relabeled, dropped, anchors, audit


def _apply_sanitization_to_graph(G, kept, rerouted_by_lobe, dropped):
    cent = set(_central_edges(G))
    rerouted = {tuple(sorted((str(u), str(v)))): str(L) for L, es in rerouted_by_lobe.items() for (u, v) in es}
    dropped = set(tuple(sorted((str(u), str(v)))) for (u, v) in dropped)

    for e in G.edges():
        if len(e) >= 2:
            u, v = str(e[0]), str(e[1])
        else:
            continue
        key = tuple(sorted((u, v)))
        if key in cent:
            ed = _edge_data(G, u, v)
            if key in dropped:
                ed["label"] = "dropped_from_central"
            elif key in rerouted:
                ed["label"] = rerouted[key]
            else:
                ed["label"] = 0

            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                for k in list(G[u][v].keys()):
                    if key in dropped:
                        G[u][v][k]["label"] = "dropped_from_central"
                    elif key in rerouted:
                        G[u][v][k]["label"] = rerouted[key]
                    else:
                        G[u][v][k]["label"] = 0
            else:
                G[u][v]["label"] = ed["label"]

    return G


def _node_xy(G, n):
    d = G.nodes[str(n)]
    for xk, yk in (("x", "y"), ("X", "Y"), ("cx", "cy"), ("pos_x", "pos_y")):
        if xk in d and yk in d:
            try:
                return float(d[xk]), float(d[yk])
            except Exception:
                pass
    if "pos" in d:
        v = d["pos"]
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str) and "," in v:
            xs, ys = v.split(",", 1)
            return float(xs), float(ys)
    return 0.0, 0.0


def qc_plot_sanitization_anchor_connectivity(graph_path, decision_pack, title_prefix="", save_path=None):
    G = nx.read_graphml(graph_path)

    rerouted_by_lobe = decision_pack.get("rerouted_by_lobe", {})
    dropped_edges = decision_pack.get("dropped_edges", [])
    anchors = decision_pack.get("anchors", [])

    rerouted_set = set()
    for L, es in rerouted_by_lobe.items():
        for u, v in es:
            rerouted_set.add(tuple(sorted((str(u), str(v)))))

    dropped_set = set(tuple(sorted((str(u), str(v)))) for (u, v) in dropped_edges)
    anchor_set = set(tuple(sorted((str(u), str(v)))) for (u, v) in anchors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for u, v in G.edges():
        uu, vv = str(u), str(v)
        x1, y1 = _node_xy(G, uu)
        x2, y2 = _node_xy(G, vv)
        key = tuple(sorted((uu, vv)))

        if key in dropped_set:
            ax.plot([x1, x2], [y1, y2], linewidth=1.6, alpha=0.85)
        elif key in rerouted_set:
            ax.plot([x1, x2], [y1, y2], linewidth=1.6, alpha=0.85)
        else:
            ax.plot([x1, x2], [y1, y2], linewidth=0.6, alpha=0.25)

    for u, v in anchors:
        x1, y1 = _node_xy(G, u)
        x2, y2 = _node_xy(G, v)
        ax.plot([x1, x2], [y1, y2], linewidth=3.0, alpha=0.95)

    ax.set_title(f"{title_prefix} | central sanitize QC")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.legend(
        handles=[
            Line2D([0], [0], lw=0.6, alpha=0.25, label="All edges"),
            Line2D([0], [0], lw=1.6, alpha=0.85, label="Relabeled / Dropped (highlight)"),
            Line2D([0], [0], lw=3.0, alpha=0.95, label="Anchors"),
        ],
        loc="best",
    )

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    plt.close(fig)


def sanitize_one_graph(
    graph_path: Path,
    out_dir: Path,
    num_anchor_edges: int,
    radius_floor_factor: float,
    geodesic_limit_mm: float,
    far_gate_mm,
    allow_drop_if_no_lobe: bool,
    write_qc_plot: bool,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        G = nx.read_graphml(graph_path)
    except Exception as e:
        raise RuntimeError(f"Failed reading {graph_path}: {e}")

    bad_before = collect_mixed_side_edges(G, allow_central_as_neutral=True)
    if bad_before:
        pd.DataFrame(
            bad_before, columns=["u", "v", "side_u", "side_v", "lobe_u", "lobe_v"]
        ).to_csv(out_dir / "_side_mismatches_before.csv", index=False)

    changed = normalize_graph_sides_from_lobes(G)

    bad_after = collect_mixed_side_edges(G, allow_central_as_neutral=True)
    if bad_after:
        pd.DataFrame(
            bad_after, columns=["u", "v", "side_u", "side_v", "lobe_u", "lobe_v"]
        ).to_csv(out_dir / "_side_mismatches_after.csv", index=False)

    kept, rerouted, dropped, anchors, audit = sanitize_central_by_anchor_connectivity(
        G,
        num_anchor_edges=num_anchor_edges,
        radius_floor_factor=radius_floor_factor,
        geodesic_limit_mm=geodesic_limit_mm,
        far_gate_mm=far_gate_mm,
        allow_drop_if_no_lobe=allow_drop_if_no_lobe,
    )

    audit_csv = out_dir / "central_sanitize_audit.csv"
    pd.DataFrame(audit).to_csv(audit_csv, index=False)

    decisions = {
        "graph": str(graph_path),
        "num_anchor_edges": int(num_anchor_edges),
        "radius_floor_factor": float(radius_floor_factor),
        "geodesic_limit_mm": float(geodesic_limit_mm),
        "far_gate_mm": (None if far_gate_mm is None else float(far_gate_mm)),
        "kept_central_count": int(len(kept)),
        "rerouted_counts": {str(k): int(len(v)) for k, v in rerouted.items()},
        "dropped_count": int(len(dropped)),
        "anchors": [(str(u), str(v)) for (u, v) in anchors],
        "side_normalized_nodes": int(changed),
        "mixed_side_edges_before": int(len(bad_before)),
        "mixed_side_edges_after": int(len(bad_after)),
    }
    decisions_json = out_dir / "central_sanitize_decisions.json"
    with open(decisions_json, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)

    G2 = _apply_sanitization_to_graph(G, kept, rerouted, dropped)
    sanitized_graphml = out_dir / "central_sanitized.graphml"
    nx.write_graphml(G2, sanitized_graphml)

    qc_png = None
    if write_qc_plot:
        qc_png = out_dir / "central_sanitize_anchor_QC.png"
        qc_plot_sanitization_anchor_connectivity(
            str(graph_path),
            dict(rerouted_by_lobe=rerouted, dropped_edges=dropped, anchors=anchors),
            title_prefix=graph_path.name,
            save_path=str(qc_png),
        )

    return {
        "graph": str(graph_path),
        "out_dir": str(out_dir),
        "sanitized_graph": str(sanitized_graphml),
        "audit_csv": str(audit_csv),
        "decisions_json": str(decisions_json),
        "qc_png": (None if qc_png is None else str(qc_png)),
        "kept": int(len(kept)),
        "dropped": int(len(dropped)),
        "rerouted": int(sum(len(v) for v in rerouted.values())),
    }


def iter_graphs(input_path: Path, recursive: bool, pattern: str):
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if recursive:
        yield from sorted(input_path.rglob(pattern))
    else:
        yield from sorted(input_path.glob(pattern))


def main():
    ap = argparse.ArgumentParser(prog="lobe_labels_sanitization_cli", description="Sanitize central lobe labels in GraphML graphs.")
    ap.add_argument("--input", required=True, type=Path, help="GraphML file or directory.")
    ap.add_argument("--output", required=True, type=Path, help="Output directory.")
    ap.add_argument("--pattern", default="*.graphml", help="Glob pattern when --input is a directory.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when --input is a directory.")

    ap.add_argument("--num-anchor-edges", type=int, default=5)
    ap.add_argument("--radius-floor-factor", type=float, default=0.7)
    ap.add_argument("--geodesic-limit-mm", type=float, default=40.0)
    ap.add_argument("--far-gate-mm", type=float, default=None)
    ap.add_argument("--no-drop-if-no-lobe", action="store_true", help="If set, keep central edges even when no lobar target exists.")
    ap.add_argument("--no-qc-plot", action="store_true", help="Disable QC PNG plot.")
    ap.add_argument("--mirror-tree", action="store_true", help="If input is a directory, mirror relative paths under output.")
    ap.add_argument("--flat", action="store_true", help="If input is a directory, write each graph into output/<stem>/ (default).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing central_sanitized.graphml outputs.")

    args = ap.parse_args()

    allow_drop_if_no_lobe = (not args.no_drop_if_no_lobe)
    write_qc_plot = (not args.no_qc_plot)

    input_path = Path(args.input)
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    graphs = list(iter_graphs(input_path, recursive=args.recursive, pattern=args.pattern))
    graphs = [g for g in graphs if g.is_file()]
    if not graphs:
        raise SystemExit("No graphml files found.")

    summary = []
    for gpath in graphs:
        if input_path.is_file():
            out_dir = out_base
        else:
            if args.mirror_tree:
                rel = gpath.parent.relative_to(input_path)
                out_dir = out_base / rel / gpath.stem
            else:
                out_dir = out_base / gpath.stem

        out_dir.mkdir(parents=True, exist_ok=True)
        sentinel = out_dir / "central_sanitized.graphml"
        if sentinel.exists() and (not args.overwrite):
            summary.append(
                {
                    "graph": str(gpath),
                    "out_dir": str(out_dir),
                    "skipped": True,
                    "reason": "exists (use --overwrite to replace)",
                }
            )
            continue

        try:
            res = sanitize_one_graph(
                gpath,
                out_dir,
                num_anchor_edges=args.num_anchor_edges,
                radius_floor_factor=args.radius_floor_factor,
                geodesic_limit_mm=args.geodesic_limit_mm,
                far_gate_mm=args.far_gate_mm,
                allow_drop_if_no_lobe=allow_drop_if_no_lobe,
                write_qc_plot=write_qc_plot,
            )
            res["skipped"] = False
            res["reason"] = ""
            summary.append(res)
            print(f"[OK] {gpath.name} -> {out_dir}", flush=True)
        except Exception as e:
            warnings.warn(f"[FAIL] {gpath}: {e}")
            summary.append({"graph": str(gpath), "out_dir": str(out_dir), "skipped": False, "reason": str(e)})

    df = pd.DataFrame(summary)
    df.to_csv(out_base / "sanitize_batch_summary.csv", index=False)
    print(f"Saved batch summary -> {out_base / 'sanitize_batch_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
