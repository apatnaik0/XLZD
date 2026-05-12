# Depth Penetration Experiments

This folder contains exploratory notebook workflows for modeling penetration depth from initial event position.

Current notebooks:

- `00_global_center_distance_vbll.ipynb`: one global model predicting `d_center = sqrt(r^2 + z_from_center^2)`
- `01_grouped_penetration_vbll.ipynb`: geometry-aware models for axial and radial penetration

Interactive HTML scripts:

- `build_tpc_explorer_2d.py`: writes a 2D cross-section explorer HTML
- `build_tpc_prediction_explorer_2d.py`: writes a 2D prediction-focused explorer HTML
- `build_tpc_explorer_3d.py`: writes a 3D TPC explorer HTML

Both notebooks:

- load a bounded subset of the raw CSV data to stay tractable
- reuse the repo's centered-coordinate conventions
- train an MLP feature extractor with a Bayesian last-layer style regression head
- show training progress with per-epoch progress bars and ETA
- produce component-level plots for diagnostics

The component grouping used in the grouped notebook is:

- top: `TPCPMTTop`, `GXe`
- bottom: `TPCPMTTBottom`, `CathodeGrid`
- side: `FieldCage`, `ICV`, `OCV`
