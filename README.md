# ML Coverage Closure Framework

## What This Is
This project is an end-to-end Machine Learning framework for functional coverage closure in verification.

It learns the relationship between stimulus knobs and coverage bin hits, then recommends targeted knob settings to close unhit bins faster than standard constrained-random loops.

Core script: `ml_cov_framework.py`

## Why It Exists
Traditional constrained-random verification can spend many redundant cycles trying to hit rare corner bins.

This framework reduces that waste by:
1. Generating labeled coverage data.
2. Training a multi-label model to predict bin-hit probability.
3. Running inverse optimization to suggest knob constraints for unhit bins.

## Key Advantage
Instead of blindly sampling a large stimulus space, the model estimates:

$$P(C_i = 1 \mid x)$$

Then optimization searches for:

$$x^* = \arg\max_{x} \sum_{i \in \mathcal{T}} P(C_i = 1 \mid x)$$

Where:
1. $C_i$ is coverage bin $i$.
2. $x$ is the stimulus knob vector.
3. $\mathcal{T}$ is the set of target (unhit) bins.

This typically cuts simulation effort for hard bins by a large factor.

## Who Benefits
1. Verification Engineers working on coverage closure.
2. DV Leads who need faster regression convergence.
3. Teams using UVM/cocotb-style randomized flows.
4. Researchers prototyping ML-guided verification.

## Turn-In Files (No Simulator Required)
1. `ml_cov_framework.py`
2. `requirements.txt`

## Environment
1. Python 3.10+
2. Packages in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## How To Use
Run from this directory.

### 1) Generate Data (Phase 1)
```bash
python3 ml_cov_framework.py datagen --seeds 2000
```
Output: `coverage_dataset.csv`

### 2) Train Model (Phase 2)
```bash
python3 ml_cov_framework.py train --model rf
```
Outputs:
1. `cov_predictor.joblib`
2. `feature_importance.csv`

### 3) Optimize for Unhit Bins (Phase 3)
```bash
python3 ml_cov_framework.py optimize --target cov_all_stress cov_cross_s0d3 cov_heavy_bp --mode loop
```
Output: optimizer report with best knobs and UVM-style constraint block.

### 4) Run Full Pipeline
```bash
python3 ml_cov_framework.py all --seeds 2000 --target cov_all_stress cov_cross_s0d3 cov_heavy_bp
```

## Outputs You Should Expect
1. Data generation progress and per-bin hit rates.
2. Validation metrics such as Hamming loss and Macro/Micro F1.
3. Feature-importance summary (RF mode).
4. Optimized knob values and predicted hit probabilities for target bins.
5. Auto-generated constraint block for targeted reruns.

## Practical Value in Verification
1. Fewer redundant simulations.
2. Faster closure of rare bins.
3. Better visibility into knob-to-coverage influence.
4. Repeatable and automatable steering loop for regressions.

## Notes
1. This README describes the simulator-free flow.
2. The script is self-contained for data generation, training, and optimization.
3. If needed, RTL/cocotb integration can be layered on top later.
