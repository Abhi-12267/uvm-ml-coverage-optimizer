#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_cov")

KNOB_NAMES = [
    "injection_rate",
    "packet_size",
    "source_id",
    "destination_id",
    "backpressure_cycles",
]

COV_BIN_NAMES = [
    "cov_min_pkt",
    "cov_max_pkt",
    "cov_loopback",
    "cov_heavy_bp",
    "cov_sat_rate",
    "cov_cross_s0d3",
    "cov_cross_s1d2",
    "cov_stress",
    "cov_low_load",
    "cov_all_stress",
]

BOUNDS: List[Tuple[float, float]] = [(1, 100), (1, 63), (0, 3), (0, 3), (0, 20)]
DEFAULT_CSV = "coverage_dataset.csv"
DEFAULT_MODEL = "cov_predictor.joblib"


def evaluate_bins(ir: int, ps: int, si: int, di: int, bp: int, rng: random.Random, noise: float) -> List[int]:
    hits = [
        int(ps == 1),
        int(ps == 63),
        int(si == di),
        int(bp >= 16),
        int(ir > 90),
        int(si == 0 and di == 3),
        int(si == 1 and di == 2),
        int(ps > 32 and bp > 10),
        int(ir < 10),
        int(ir > 80 and ps > 48 and bp > 12),
    ]
    return [h if (h == 0 or rng.random() >= noise) else 0 for h in hits]


def sample_knobs(rng: random.Random, biased_ratio: float) -> Tuple[int, int, int, int, int]:
    if rng.random() < biased_ratio:
        ir = rng.choice([rng.randint(1, 9), rng.randint(91, 100)])
        ps = rng.choice([1, 63, rng.randint(33, 63)])
        bp = rng.randint(13, 20)
    else:
        ir, ps, bp = rng.randint(1, 100), rng.randint(1, 63), rng.randint(0, 20)
    return ir, ps, rng.randint(0, 3), rng.randint(0, 3), bp


def run_datagen(seeds: int, biased_ratio: float, noise: float, out_csv: str) -> pd.DataFrame:
    rng = random.Random(0)
    rows = []
    totals = {b: 0 for b in COV_BIN_NAMES}
    for seed in range(seeds):
        rng.seed(seed)
        ir, ps, si, di, bp = sample_knobs(rng, biased_ratio)
        bins = evaluate_bins(ir, ps, si, di, bp, rng, noise)
        row = {
            "seed": seed,
            "injection_rate": ir,
            "packet_size": ps,
            "source_id": si,
            "destination_id": di,
            "backpressure_cycles": bp,
        }
        for b, h in zip(COV_BIN_NAMES, bins):
            row[b] = h
            totals[b] += h
        rows.append(row)
        if (seed + 1) % 500 == 0:
            log.info("datagen %d/%d", seed + 1, seeds)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info("phase1 done: %s shape=%s", out_csv, df.shape)
    log.info("bin hit rates: %s", {b: round(100 * totals[b] / seeds, 2) for b in COV_BIN_NAMES})
    return df


def sample_weights(y: np.ndarray) -> np.ndarray:
    sw = np.ones(y.shape[0], dtype=np.float32)
    for i in range(y.shape[1]):
        classes = np.unique(y[:, i])
        if len(classes) < 2:
            continue
        w = compute_class_weight("balanced", classes=classes, y=y[:, i])
        for c, wt in zip(classes, w):
            sw[y[:, i] == c] *= wt
    return sw / sw.mean()


def build_pipeline(model: str) -> Pipeline:
    if model == "rf":
        base = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=1,
            random_state=42,
        )
        return Pipeline([("model", MultiOutputClassifier(base, n_jobs=1))])
    base = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", MultiOutputClassifier(base, n_jobs=1))])


def run_train(csv_path: str, model: str, out_model: str) -> Pipeline:
    df = pd.read_csv(csv_path)
    X = df[KNOB_NAMES].values.astype(np.float32)
    y = df[COV_BIN_NAMES].values.astype(np.int32)
    comp = y.dot(1 << np.arange(y.shape[1]))
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=comp, random_state=42)
    except ValueError:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(model)
    pipe.fit(X_tr, y_tr, model__sample_weight=sample_weights(y_tr))
    y_hat = pipe.predict(X_va)
    log.info("phase2: Hamming=%.4f MacroF1=%.4f MicroF1=%.4f",
             hamming_loss(y_va, y_hat),
             f1_score(y_va, y_hat, average="macro", zero_division=0),
             f1_score(y_va, y_hat, average="micro", zero_division=0))
    for line in classification_report(y_va, y_hat, target_names=COV_BIN_NAMES, zero_division=0).split("\n"):
        if line.strip():
            log.info(line)
    if model == "rf":
        multi = pipe.named_steps["model"]
        fi = []
        for b, est in zip(COV_BIN_NAMES, multi.estimators_):
            vals = est.feature_importances_
            fi.append({"coverage_bin": b, **{k: round(float(v), 4) for k, v in zip(KNOB_NAMES, vals)}})
            log.info("%s dominant=%s", b, KNOB_NAMES[int(vals.argmax())])
        pd.DataFrame(fi).set_index("coverage_bin").to_csv(Path(csv_path).parent / "feature_importance.csv")
    joblib.dump(pipe, out_model)
    log.info("phase2 done: %s", out_model)
    return pipe


def proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    X2 = X.reshape(1, -1) if X.ndim == 1 else X
    raw = pipeline.predict_proba(X2)
    out = np.column_stack([a[:, 1] for a in raw])
    return out[0] if X.ndim == 1 else out


@dataclass
class OptResult:
    targets: List[str]
    knobs: Dict[str, int]
    probs: Dict[str, float]
    objective: float
    strategy: str = "de"
    topk: List[Dict[str, int]] = field(default_factory=list)

    def joint(self) -> float:
        return float(np.prod([self.probs[t] for t in self.targets]))

    def uvm(self) -> str:
        lines = ["constraint c_ml_steered {"]
        for i, (k, v) in enumerate(self.knobs.items()):
            lo, hi = BOUNDS[i]
            margin = max(1, int(round((hi - lo) * 0.10)))
            a, b = max(int(lo), v - margin), min(int(hi), v + margin)
            lines.append(f"    {k} inside {{[{a}:{b}]}};")
        lines.append("}")
        return "\n".join(lines)


def optimize_once(pipeline: Pipeline, targets: List[str], lam: float, pop: int, iters: int, grid: int) -> OptResult:
    t_idx = [COV_BIN_NAMES.index(t) for t in targets]
    nt_idx = [i for i in range(len(COV_BIN_NAMES)) if i not in t_idx]

    def obj(x: np.ndarray) -> float:
        p = proba(pipeline, np.round(x))
        return -(p[t_idx].sum() - lam * p[nt_idx].sum())

    log.info("phase3: DE pop=%d maxiter=%d", pop, iters)
    res = differential_evolution(obj, BOUNDS, popsize=pop, maxiter=iters, seed=42,
                                 mutation=0.7, recombination=0.9, tol=1e-5,
                                 polish=True, init="latinhypercube")
    x = np.round(res.x).astype(int)
    p = proba(pipeline, x.astype(float))
    out = OptResult(
        targets=targets,
        knobs={k: int(v) for k, v in zip(KNOB_NAMES, x)},
        probs={b: float(v) for b, v in zip(COV_BIN_NAMES, p)},
        objective=float(res.fun),
        strategy="de",
    )
    log.info("phase3 result: joint=%.4f objective=%.4f", out.joint(), out.objective)
    for k, v in out.knobs.items():
        log.info("  %s=%d", k, v)
    for b in targets:
        log.info("  target %s: %.4f", b, out.probs[b])
    log.info("%s", out.uvm())
    return out


def run_optimize(model_path: str, targets: List[str], mode: str, lam: float, pop: int, iters: int, grid: int, max_iter: int):
    pipe = joblib.load(model_path)
    if mode == "single":
        optimize_once(pipe, targets, lam, pop, iters, grid)
        return
    unhit = list(targets)
    for i in range(max_iter):
        log.info("loop iter %d unhit=%s", i + 1, unhit)
        r = optimize_once(pipe, unhit, lam, pop, iters, grid)
        newly = [b for b in unhit if r.probs[b] > 0.5]
        log.info("newly hit=%s", newly)
        unhit = [b for b in unhit if b not in newly]
        if not unhit:
            log.info("closure achieved in %d iterations", i + 1)
            return
        if not newly:
            log.warning("no progress, stop")
            return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file ML coverage closure framework")
    s = p.add_subparsers(dest="cmd", required=True)

    d = s.add_parser("datagen")
    d.add_argument("--seeds", type=int, default=2000)
    d.add_argument("--biased-ratio", type=float, default=0.25)
    d.add_argument("--noise", type=float, default=0.02)
    d.add_argument("--out", type=str, default=DEFAULT_CSV)

    t = s.add_parser("train")
    t.add_argument("--csv", type=str, default=DEFAULT_CSV)
    t.add_argument("--model", type=str, default="rf", choices=["rf", "mlp"])
    t.add_argument("--out", type=str, default=DEFAULT_MODEL)

    o = s.add_parser("optimize")
    o.add_argument("--model", type=str, default=DEFAULT_MODEL)
    o.add_argument("--target", nargs="+", default=["cov_all_stress", "cov_cross_s0d3", "cov_heavy_bp"])
    o.add_argument("--mode", type=str, default="single", choices=["single", "loop"])
    o.add_argument("--lam", type=float, default=0.0)
    o.add_argument("--de-popsize", type=int, default=15)
    o.add_argument("--de-maxiter", type=int, default=500)
    o.add_argument("--grid-points", type=int, default=10)
    o.add_argument("--max-iter", type=int, default=10)

    a = s.add_parser("all")
    a.add_argument("--seeds", type=int, default=2000)
    a.add_argument("--csv", type=str, default=DEFAULT_CSV)
    a.add_argument("--model", type=str, default="rf", choices=["rf", "mlp"])
    a.add_argument("--model-out", type=str, default=DEFAULT_MODEL)
    a.add_argument("--target", nargs="+", default=["cov_all_stress", "cov_cross_s0d3", "cov_heavy_bp"])
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "datagen":
        run_datagen(a.seeds, a.biased_ratio, a.noise, a.out)
    elif a.cmd == "train":
        run_train(a.csv, a.model, a.out)
    elif a.cmd == "optimize":
        run_optimize(a.model, a.target, a.mode, a.lam, a.de_popsize, a.de_maxiter, a.grid_points, a.max_iter)
    elif a.cmd == "all":
        run_datagen(a.seeds, 0.25, 0.02, a.csv)
        run_train(a.csv, a.model, a.model_out)
        run_optimize(a.model_out, a.target, "loop", 0.0, 15, 500, 10, 10)


if __name__ == "__main__":
    main()
