# SR from CA/MM (poset-based)

This repo shows SR kinematics emerging from **Conscious Agents / Markov Matrix** with only
one-step-per-tick causal updates, a partial order (no backward edges), and counting.
The full SR suite lives in `src/` and prints PASS/FAIL JSON.

Quick run:
```bash
cd src
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast
cat sr_all_summary.json    # expect "ALL_PASS": true
See src/README_SR.md for per-test CLI, math, and intuition.

Minimal requirements
printf "numpy\n" > requirements.txt

GitHub Actions CI workflow
mkdir -p .github/workflows
cat > .github/workflows/sr.yml << 'YAML'
name: SR suite (fast)
on:
push: { branches: [ main ] }
pull_request: { branches: [ main ] }
jobs:
sr-fast:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
with: { python-version: '3.12' }
- name: Install deps
run: |
python -m pip install -U pip
pip install -r requirements.txt
- name: Run SR fast battery
working-directory: src
run: |
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast
- name: Verify ALL_PASS
working-directory: src
run: |
python - << 'PY'
import json,sys
with open('sr_all_summary.json') as f:
j=json.load(f)
ok=j.get("ALL_PASS", False)
print("ALL_PASS:", ok)
sys.exit(0 if ok else 1)
PY
YAML

Optional explainer page (nice for GitHub browsing)
mkdir -p docs
cat > docs/SR_Explainer.md << 'MD'

SR from CA/MM — One-page intuition
See src/README_SR.md for full CLI + math. This page summarizes the ideas.

Light-cone: one step/tick ⇒ causal speed ĉ=1.

Dilation: Alexandrov counts ratio ≈ √(1−v²).

Length: Δ(E)=N(p−,E)−N(E,p+) defines simultaneity; projection gives L′≈L0√(1−v²).

Interval: counts → invariant s² without assuming a metric.

Velocity composition: rapidities add.

Simultaneity: Δ signs flip across frames.

Isotropy (stats): random micro-directions ⇒ near-circular covariances.

Dimension signal: order fractions distinguish 1+1 vs 2+1.
