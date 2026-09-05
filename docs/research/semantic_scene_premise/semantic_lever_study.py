"""Which knobs set the semantic scene detector's scene count? (owner C1/C2/C6, 2026-09-05)

Extract the engine's own features ONCE per file, then re-run only the segmentation stage under
three boundary logics and report how the scene count responds to each knob:

  A. shipped: unconstrained Ward clustering (global) -> cut at label flips -> snap -> min_dur merge
  B. same features, Ward clustering CONSTRAINED to adjacent samples (temporal chain) -> contiguous
     clusters ARE the segments -> snap -> min_dur merge
  C. change-point on the same features: cosine distance between the mean of the W seconds before
     and after each point; boundary where it exceeds mean + k*std and is a local maximum -> snap ->
     min_dur merge

Every arm finishes with the engine's own _snap_to_silence / _smart_merge / _forced_cleanup /
_ensure_timeline_coverage, so the boundary logic is the only difference between arms. Two downstream
knobs are ALSO swept under the shipped logic (min_duration, snap_window) because the first version of
this study held them fixed and wrongly declared them neutral.

No accuracy is measured (owner C5). Only counts: clusters, raw boundaries, final scenes.

Usage (from the repo root; -X utf8 because the engine prints a non-cp1252 arrow):
    python -X utf8 docs/research/semantic_scene_premise/semantic_lever_study.py WAV [WAV ...] OUT.json
The LAST argument is the output JSON.
Versions used for the committed results: sklearn 1.6.1, numpy 1.26.4, scipy 1.17.0.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from whisperjav.vendor import semantic_audio_clustering as sac  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.sparse import diags  # noqa: E402

THRESHOLDS = [5, 10, 18, 30, 50, 90, 200, 500, 1000, 3000]
FINE_THRESHOLDS = list(range(5, 41))          # the exposed slider is 5-30; go a little past it
MIN_DURS = [5, 10, 20, 30, 60]
SNAP_WINDOWS = [1.0, 2.0, 5.0, 8.0, 12.0]     # 5.0 is the default; presets use 2-8
WINDOWS_S = [5, 10, 20, 30]
KS = [1.0, 1.5, 2.0, 3.0]


def features_for(path: Path):
    cfg = sac.SegmentationConfig()
    ext = sac.StreamFeatureExtractor(cfg, logger=None)
    t0 = time.time()
    feats, times, dur = ext.extract(str(path))
    cls = sac.AdaptiveClassifier(cfg, None)
    cls.calibrate(feats)
    floor = cls.stats["rms_base"] * cfg.silence_threshold_multiplier
    return cfg, feats, times, dur, floor, time.time() - t0


def prepared(cfg, feats, times):
    fs = median_filter(feats, size=(1, cfg.smoothing_window))
    step = int(cfg.fps * 0.5)                  # 15 hops x 512 / 16000 = 0.48 s per vector
    X = StandardScaler().fit_transform(fs[:, ::step].T)
    return X, times[::step], step * cfg.hop_length / cfg.sample_rate


def finish(boundaries, feats, times, dur, floor, min_dur=20.0, snap_window=5.0):
    seg = sac.SemanticSegmenter(sac.SegmentationConfig(min_duration=min_dur, snap_window=snap_window), None)
    b = seg._snap_to_silence(boundaries, feats, times, floor)
    s = seg._smart_merge(b, feats, times)
    s = seg._forced_cleanup(s)
    s = seg._ensure_timeline_coverage(s, dur)
    return len(s)


def labels_to_boundaries(labels, Xt, dur):
    return [0.0] + [Xt[i] for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [dur]


def run_stats(labels):
    runs = np.diff(np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True]))
    return {"runs": int(len(runs)), "single_sample_runs": int((runs == 1).sum()),
            "median_run_samples": float(np.median(runs)), "max_run_samples": int(runs.max())}


def labels_a(X, thr):
    return AgglomerativeClustering(n_clusters=None, distance_threshold=thr, linkage="ward").fit_predict(X)


def labels_b(X, thr):
    n = len(X)
    conn = diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n)).tocsr()
    return AgglomerativeClustering(n_clusters=None, distance_threshold=thr, linkage="ward",
                                   connectivity=conn).fit_predict(X)


def boundaries_c(X, Xt, dur, win_s, k, step_s):
    w = max(1, int(round(win_s / step_s)))
    n = len(X)
    d = np.zeros(n)
    for i in range(w, n - w):
        a, b = X[i - w:i].mean(axis=0), X[i:i + w].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        d[i] = 1.0 - float(a @ b / (na * nb)) if na and nb else 0.0
    interior = d[w:n - w]
    thr = interior.mean() + k * interior.std()      # NOTE: relative to this file's distribution
    peaks = [i for i in range(w, n - w) if d[i] > thr and d[i] >= d[max(0, i - w):i + w + 1].max()]
    return [0.0] + [Xt[i] for i in peaks] + [dur], len(peaks)


def main(paths, out_path):
    out = {"versions": {}}
    import sklearn, scipy
    out["versions"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    for p in paths:
        p = Path(p)
        cfg, feats, times, dur, floor, t_ext = features_for(p)
        X, Xt, step_s = prepared(cfg, feats, times)
        rec = {"duration_s": round(dur, 1), "vectors": int(len(X)), "vector_period_s": round(step_s, 3),
               "feature_extraction_s": round(t_ext, 1), "silence_floor": floor,
               "A_unconstrained": [], "B_temporal_constrained": [], "A_fine": [], "B_fine": [],
               "A_min_duration_sweep_thr18": [], "A_snap_window_sweep": [], "A_label_runs": [], "C_changepoint": []}
        pr = lambda *a: print(p.name, *a, flush=True)  # noqa: E731
        for thr in THRESHOLDS:
            la = labels_a(X, thr); ba = labels_to_boundaries(la, Xt, dur)
            lb = labels_b(X, thr); bb = labels_to_boundaries(lb, Xt, dur)
            fa, fb = finish(ba, feats, times, dur, floor), finish(bb, feats, times, dur, floor)
            rec["A_unconstrained"].append({"threshold": thr, "clusters": int(la.max() + 1), "raw_boundaries": len(ba) - 2, "scenes": fa})
            rec["B_temporal_constrained"].append({"threshold": thr, "segments": int(lb.max() + 1), "raw_boundaries": len(bb) - 2, "scenes": fb})
            if thr in (18, 90):
                rec["A_label_runs"].append({"threshold": thr, **run_stats(la)})
            pr(f"thr={thr:>5}: A clusters={la.max()+1:>5} raw={len(ba)-2:>5} scenes={fa:>4} | B segments={lb.max()+1:>5} scenes={fb:>4}")
        for thr in FINE_THRESHOLDS:
            fa = finish(labels_to_boundaries(labels_a(X, thr), Xt, dur), feats, times, dur, floor)
            fb = finish(labels_to_boundaries(labels_b(X, thr), Xt, dur), feats, times, dur, floor)
            rec["A_fine"].append({"threshold": thr, "scenes": fa}); rec["B_fine"].append({"threshold": thr, "scenes": fb})
            pr(f"fine thr={thr:>3}: A scenes={fa:>4} B scenes={fb:>4}")
        ba18 = labels_to_boundaries(labels_a(X, 18), Xt, dur)
        for md in MIN_DURS:
            fa = finish(ba18, feats, times, dur, floor, min_dur=float(md))
            rec["A_min_duration_sweep_thr18"].append({"min_duration": md, "scenes": fa}); pr(f"thr=18 min_dur={md}: scenes={fa}")
        for thr in (18, 50):
            b = labels_to_boundaries(labels_a(X, thr), Xt, dur)
            for sw in SNAP_WINDOWS:
                fa = finish(b, feats, times, dur, floor, snap_window=sw)
                rec["A_snap_window_sweep"].append({"threshold": thr, "snap_window": sw, "scenes": fa}); pr(f"thr={thr} snap_window={sw}: scenes={fa}")
        for w in WINDOWS_S:
            for k in KS:
                b, peaks = boundaries_c(X, Xt, dur, w, k, step_s)
                fc = finish(b, feats, times, dur, floor)
                rec["C_changepoint"].append({"window_s": w, "k_sigma": k, "peaks": peaks, "scenes": fc}); pr(f"C win={w}s k={k}: peaks={peaks} scenes={fc}")
        out[p.name] = rec
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: semantic_lever_study.py WAV [WAV ...] OUT.json")
    main(sys.argv[1:-1], sys.argv[-1])
