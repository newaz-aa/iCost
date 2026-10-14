# iCost 0.3.0 release checklist

## 1. Verify locally

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

Optional isolated smoke test:

```bash
python -m venv .venv-release
# Windows: .venv-release\\Scripts\\activate
# Linux/macOS: source .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/icost-0.3.0-py3-none-any.whl
python -c "from icost import iCost, __version__; print(__version__)"
```

## 2. Upload to TestPyPI first (recommended)

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation from TestPyPI in a clean environment.

## 3. Publish to PyPI

```bash
python -m twine upload dist/*
```

## 4. GitHub release

- Commit the source changes.
- Tag the release as `v0.3.0`.
- Create a GitHub release using the `CHANGELOG.md` notes.
- Optionally attach the wheel and source distribution from `dist/`.

## Release highlights

- Cross-fitted Gini-iCost complexity estimation.
- Recommended fixed defaults `(cfo, cfp, cfs, cfb) = (0.10, 0.30, 0.50, 0.70) * IR`.
- Gini structural defaults: 3-fold cross-fitting, depth 3, leaf size 5, `tau_min=0.80`, `tau_maj=0.20`.
- Native multiclass one-vs-rest support.
- `cs_multiplier` for conventional class-level CSL.
- Generalized Neighbor-iCost categorization for arbitrary `n_neighbors`.
- Exploratory Cluster-iCost with class-specific KMeans regions and fixed distance thresholds `(0.25, 0.50, 0.75)`.
- scikit-learn classifier tagging and conditional response-method compatibility fixes.
- Backward-compatible aliases for iCost 0.2.x parameter names.
