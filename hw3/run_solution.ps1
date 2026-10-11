# Run HW3 solution when local Python DLLs are blocked by Windows App Control.
# Requires Docker Desktop.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

docker run --rm `
  -v "${ProjectRoot}:/work" `
  -w /work `
  python:3.12-slim `
  bash -c "pip install -q pandas numpy scikit-learn matplotlib seaborn && python hw3_solution.py"
