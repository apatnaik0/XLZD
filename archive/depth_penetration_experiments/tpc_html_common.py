from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from depth_vbll import (
    FIXED_TPC_R_MAX,
    FIXED_TPC_Z_CENTER,
    FIXED_TPC_Z_MAX,
    load_depth_dataset,
)

COMPONENT_GROUP_ORDER = ("top", "bottom", "side")
PLOTLY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "prepare_resum_data.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root.")


def load_plot_dataframe(
    data_dir: str | Path,
    *,
    max_rows_per_component: int = 25000,
    max_points_per_component: int = 3000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, str]]:
    dataset = load_depth_dataset(data_dir, max_rows_per_component=max_rows_per_component)
    df = dataset.df.copy()
    df["z_centered_final"] = df["z"].to_numpy(dtype=float) - float(dataset.z_center)

    rng = np.random.default_rng(random_state)
    sampled_parts: list[pd.DataFrame] = []
    for component, part in df.groupby("source_component", sort=True):
        if len(part) > max_points_per_component:
            idx = rng.choice(len(part), size=max_points_per_component, replace=False)
            sampled_parts.append(part.iloc[idx].copy())
        else:
            sampled_parts.append(part.copy())
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled, dataset.component_groups


def component_color_map(components: Sequence[str]) -> dict[str, str]:
    ordered = list(components)
    return {component: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, component in enumerate(ordered)}


def group_to_components(component_groups: dict[str, str]) -> dict[str, list[str]]:
    mapping = {"all": sorted(component_groups)}
    for group in COMPONENT_GROUP_ORDER:
        mapping[group] = [component for component, grp in sorted(component_groups.items()) if grp == group]
    return mapping


def visibility_buttons(trace_component_meta: Sequence[str | None], component_groups: dict[str, str]) -> list[dict]:
    groups = group_to_components(component_groups)
    buttons: list[dict] = []
    for label, allowed_components in [
        ("Show All", groups["all"]),
        ("Top Only", groups["top"]),
        ("Bottom Only", groups["bottom"]),
        ("Side Only", groups["side"]),
    ]:
        visible = []
        for component in trace_component_meta:
            if component is None:
                visible.append(True)
            else:
                visible.append(component in allowed_components)
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}],
            }
        )
    return buttons


def cylinder_wireframe_xyz(
    *,
    r_max: float = FIXED_TPC_R_MAX,
    z_max: float = FIXED_TPC_Z_MAX,
    n_theta: int = 120,
    n_vertical: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    x_circle = r_max * np.cos(theta)
    y_circle = r_max * np.sin(theta)
    traces = [
        (x_circle, y_circle, np.full_like(theta, z_max)),
        (x_circle, y_circle, np.full_like(theta, -z_max)),
    ]
    for angle in np.linspace(0.0, 2.0 * np.pi, n_vertical, endpoint=False):
        x = np.array([r_max * np.cos(angle), r_max * np.cos(angle)])
        y = np.array([r_max * np.sin(angle), r_max * np.sin(angle)])
        z = np.array([-z_max, z_max])
        traces.append((x, y, z))
    return traces


def cross_section_bounds(
    *,
    r_max: float = FIXED_TPC_R_MAX,
    z_max: float = FIXED_TPC_Z_MAX,
) -> dict[str, list[float]]:
    return {
        "x": [0.0, r_max, r_max, 0.0, 0.0],
        "y": [-z_max, -z_max, z_max, z_max, -z_max],
    }


def output_path_or_default(path: str | Path | None, default_name: str) -> Path:
    if path is not None:
        return Path(path)
    return find_repo_root() / "depth_penetration_experiments" / "artifacts" / default_name


def require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for the interactive HTML explorers. "
            "Install it with `pip install plotly` or `pip install -r requirements.txt`."
        ) from exc
    return go, make_subplots
