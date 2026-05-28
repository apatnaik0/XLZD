from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from depth_vbll import predict_checkpoint_on_dataframe
from tpc_html_common import (
    FIXED_TPC_R_MAX,
    FIXED_TPC_Z_MAX,
    cylinder_wireframe_xyz,
    find_repo_root,
    group_to_components,
    load_plot_dataframe,
    output_path_or_default,
    require_plotly,
)


METRIC_SPECS = [
    {
        "label": "True d_center",
        "column": "d_center",
        "colorscale": "Viridis_r",
        "colorbar_title": "true d_center",
    },
    {
        "label": "Predicted d_center",
        "column": "pred_mean",
        "colorscale": "Viridis_r",
        "colorbar_title": "predicted d_center",
    },
    {
        "label": "Predicted Std",
        "column": "pred_std",
        "colorscale": "Magma",
        "colorbar_title": "predicted std",
    },
    {
        "label": "Absolute Error",
        "column": "pred_abs_error",
        "colorscale": "Plasma",
        "colorbar_title": "absolute error",
    },
    {
        "label": "Conservative Danger",
        "column": "pred_danger_score",
        "colorscale": "Cividis_r",
        "colorbar_title": "pred_mean - pred_std",
    },
]


def _scene_ranges(df) -> tuple[dict, dict]:
    x_initial = df["sx"].to_numpy(dtype=float)
    y_initial = df["sy"].to_numpy(dtype=float)
    z_initial = df["sz_centered"].to_numpy(dtype=float)
    x_final = df["x"].to_numpy(dtype=float)
    y_final = df["y"].to_numpy(dtype=float)
    z_final = df["z_centered_final"].to_numpy(dtype=float)

    xy_initial = max(
        float(FIXED_TPC_R_MAX),
        float(np.nanmax(np.abs(x_initial))),
        float(np.nanmax(np.abs(y_initial))),
    )
    z_initial_abs = max(float(FIXED_TPC_Z_MAX), float(np.nanmax(np.abs(z_initial))))
    xy_final = max(
        float(FIXED_TPC_R_MAX),
        float(np.nanmax(np.abs(x_final))),
        float(np.nanmax(np.abs(y_final))),
    )
    z_final_abs = max(float(FIXED_TPC_Z_MAX), float(np.nanmax(np.abs(z_final))))

    scene_initial = {
        "xaxis_title": "sx",
        "yaxis_title": "sy",
        "zaxis_title": "sz_centered",
        "aspectmode": "cube",
        "xaxis": {"range": [-xy_initial * 1.05, xy_initial * 1.05]},
        "yaxis": {"range": [-xy_initial * 1.05, xy_initial * 1.05]},
        "zaxis": {"range": [-z_initial_abs * 1.05, z_initial_abs * 1.05]},
        "camera": {"eye": {"x": 1.6, "y": 1.45, "z": 1.15}},
    }
    scene_final = {
        "xaxis_title": "x",
        "yaxis_title": "y",
        "zaxis_title": "z_centered_final",
        "aspectmode": "cube",
        "xaxis": {"range": [-xy_final * 1.05, xy_final * 1.05]},
        "yaxis": {"range": [-xy_final * 1.05, xy_final * 1.05]},
        "zaxis": {"range": [-z_final_abs * 1.05, z_final_abs * 1.05]},
        "camera": {"eye": {"x": 1.6, "y": 1.45, "z": 1.15}},
    }
    return scene_initial, scene_final


def _metric_arrays(df, trace_indices: list[int], metric_column: str, colorscale: str, colorbar_title: str) -> dict:
    color_payload = []
    colorscale_payload = []
    cmin_payload = []
    cmax_payload = []
    showscale_payload = []
    colorbar_title_payload = []

    cmin = float(np.nanmin(df[metric_column].to_numpy(dtype=float)))
    cmax = float(np.nanmax(df[metric_column].to_numpy(dtype=float)))

    for _ in trace_indices[:-1]:
        color_payload.append(df.iloc[0:0][metric_column].to_numpy(dtype=float))
        colorscale_payload.append(colorscale)
        cmin_payload.append(cmin)
        cmax_payload.append(cmax)
        showscale_payload.append(False)
        colorbar_title_payload.append(colorbar_title)

    dummy_color = np.array([cmin, cmax], dtype=float)
    color_payload.append(dummy_color)
    colorscale_payload.append(colorscale)
    cmin_payload.append(cmin)
    cmax_payload.append(cmax)
    showscale_payload.append(True)
    colorbar_title_payload.append(colorbar_title)

    return {
        "marker.color": color_payload,
        "marker.colorscale": colorscale_payload,
        "marker.cmin": cmin_payload,
        "marker.cmax": cmax_payload,
        "marker.showscale": showscale_payload,
        "marker.colorbar.title.text": colorbar_title_payload,
    }


def build_figure(df, component_groups):
    go, make_subplots = require_plotly()
    components = sorted(df["source_component"].unique())
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Initial Positions Colored By Prediction Metric", "Final Positions Colored By Prediction Metric"),
        horizontal_spacing=0.03,
    )

    df = df.copy()
    df["pred_danger_score"] = df["pred_mean"] - df["pred_std"]

    trace_component_meta: list[str | None] = []
    metric_trace_indices: list[int] = []
    component_row_map: dict[str, np.ndarray] = {}

    default_metric = METRIC_SPECS[1]

    for component in components:
        part = df[df["source_component"] == component].copy()
        metric_values = part[default_metric["column"]].to_numpy(dtype=float)
        component_row_map[component] = part.index.to_numpy(dtype=int)
        hover = np.column_stack(
            [
                part["source_component"],
                part["component_group"],
                np.round(part["d_center"], 2),
                np.round(part["pred_mean"], 2),
                np.round(part["pred_std"], 2),
                np.round(part["pred_abs_error"], 2),
                np.round(part["pred_danger_score"], 2),
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=part["sx"],
                y=part["sy"],
                z=part["sz_centered"],
                mode="markers",
                marker={
                    "size": 3,
                    "opacity": 0.55,
                    "color": metric_values,
                    "colorscale": default_metric["colorscale"],
                    "showscale": False,
                },
                name=component,
                legendgroup=component,
                showlegend=True,
                customdata=hover,
                hovertemplate=(
                    "component=%{customdata[0]}<br>"
                    "group=%{customdata[1]}<br>"
                    "sx=%{x:.2f}<br>"
                    "sy=%{y:.2f}<br>"
                    "sz_centered=%{z:.2f}<br>"
                    "true d_center=%{customdata[2]}<br>"
                    "predicted d_center=%{customdata[3]}<br>"
                    "predicted std=%{customdata[4]}<br>"
                    "absolute error=%{customdata[5]}<br>"
                    "danger score=%{customdata[6]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        trace_component_meta.append(component)
        metric_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter3d(
                x=part["x"],
                y=part["y"],
                z=part["z_centered_final"],
                mode="markers",
                marker={
                    "size": 3,
                    "opacity": 0.55,
                    "color": metric_values,
                    "colorscale": default_metric["colorscale"],
                    "showscale": False,
                },
                name=component,
                legendgroup=component,
                showlegend=False,
                customdata=hover,
                hovertemplate=(
                    "component=%{customdata[0]}<br>"
                    "group=%{customdata[1]}<br>"
                    "x=%{x:.2f}<br>"
                    "y=%{y:.2f}<br>"
                    "z_centered_final=%{z:.2f}<br>"
                    "true d_center=%{customdata[2]}<br>"
                    "predicted d_center=%{customdata[3]}<br>"
                    "predicted std=%{customdata[4]}<br>"
                    "absolute error=%{customdata[5]}<br>"
                    "danger score=%{customdata[6]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        trace_component_meta.append(component)
        metric_trace_indices.append(len(fig.data) - 1)

    for x, y, z in cylinder_wireframe_xyz():
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "black", "width": 3},
                name="TPC bounds",
                legendgroup="TPC bounds",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        trace_component_meta.append(None)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "black", "width": 3},
                name="TPC bounds",
                legendgroup="TPC bounds",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        trace_component_meta.append(None)

    default_cmin = float(np.nanmin(df[default_metric["column"]].to_numpy(dtype=float)))
    default_cmax = float(np.nanmax(df[default_metric["column"]].to_numpy(dtype=float)))
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="markers",
            marker={
                "size": 0.1,
                "opacity": 0.0,
                "color": np.array([default_cmin, default_cmax], dtype=float),
                "colorscale": default_metric["colorscale"],
                "cmin": default_cmin,
                "cmax": default_cmax,
                "showscale": True,
                "colorbar": {
                    "title": default_metric["colorbar_title"],
                    "x": 1.02,
                    "xanchor": "left",
                    "y": 0.5,
                    "len": 0.7,
                    "thickness": 16,
                },
            },
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    trace_component_meta.append(None)
    metric_trace_indices.append(len(fig.data) - 1)

    filter_map = group_to_components(component_groups)
    filter_buttons = []
    for label, key in [
        ("Show All", "all"),
        ("Top Only", "top"),
        ("Bottom Only", "bottom"),
        ("Side Only", "side"),
    ]:
        allowed = set(filter_map[key])
        visible = []
        for component in trace_component_meta:
            if component is None:
                visible.append(True)
            else:
                visible.append(component in allowed)
        filter_buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}],
            }
        )

    metric_buttons = []
    for spec in METRIC_SPECS:
        color_payload = []
        colorscale_payload = []
        cmin_payload = []
        cmax_payload = []
        showscale_payload = []
        colorbar_title_payload = []

        cmin = float(np.nanmin(df[spec["column"]].to_numpy(dtype=float)))
        cmax = float(np.nanmax(df[spec["column"]].to_numpy(dtype=float)))

        for component in components:
            rows = component_row_map[component]
            values = df.loc[rows, spec["column"]].to_numpy(dtype=float)
            color_payload.extend([values, values])
            colorscale_payload.extend([spec["colorscale"], spec["colorscale"]])
            cmin_payload.extend([cmin, cmin])
            cmax_payload.extend([cmax, cmax])
            showscale_payload.extend([False, False])
            colorbar_title_payload.extend([spec["colorbar_title"], spec["colorbar_title"]])

        color_payload.append(np.array([cmin, cmax], dtype=float))
        colorscale_payload.append(spec["colorscale"])
        cmin_payload.append(cmin)
        cmax_payload.append(cmax)
        showscale_payload.append(True)
        colorbar_title_payload.append(spec["colorbar_title"])

        metric_buttons.append(
            {
                "label": spec["label"],
                "method": "restyle",
                "args": [
                    {
                        "marker.color": color_payload,
                        "marker.colorscale": colorscale_payload,
                        "marker.cmin": cmin_payload,
                        "marker.cmax": cmax_payload,
                        "marker.showscale": showscale_payload,
                        "marker.colorbar.title.text": colorbar_title_payload,
                    },
                    metric_trace_indices,
                ],
            }
        )

    scene_initial, scene_final = _scene_ranges(df)
    fig.update_layout(
        title="XLZD TPC 3D Prediction Explorer",
        template="plotly_white",
        height=900,
        width=1850,
        legend={
            "groupclick": "togglegroup",
            "x": 1.14,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
        },
        scene=scene_initial,
        scene2=scene_final,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": filter_buttons,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
            },
            {
                "type": "dropdown",
                "buttons": metric_buttons,
                "x": 0.34,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
            },
        ],
        margin={"l": 10, "r": 260, "t": 115, "b": 10},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prediction-focused 3D TPC explorer HTML.")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw CSV data directory. Defaults to <repo>/data.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--max-rows-per-component", type=int, default=25000, help="Rows to load per component before downsampling.")
    parser.add_argument("--max-points-per-component", type=int, default=2500, help="Points per component to plot.")
    parser.add_argument(
        "--prediction-checkpoint",
        type=str,
        default=None,
        help="Optional VBLL checkpoint path. Defaults to the global center-distance model checkpoint.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_path = output_path_or_default(args.output, "tpc_penetration_prediction_explorer_3d.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_checkpoint = repo_root / "depth_penetration_experiments" / "artifacts" / "global_center_distance" / "vbll_regressor.pt"
    checkpoint_path = Path(args.prediction_checkpoint) if args.prediction_checkpoint else default_checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Prediction checkpoint not found: {checkpoint_path}. "
            "Run the global penetration notebook first or pass --prediction-checkpoint."
        )

    df, component_groups = load_plot_dataframe(
        data_dir,
        max_rows_per_component=args.max_rows_per_component,
        max_points_per_component=args.max_points_per_component,
    )
    df = predict_checkpoint_on_dataframe(df, checkpoint_path=checkpoint_path)
    fig = build_figure(df, component_groups)
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "scrollZoom": True},
    )
    print(f"Wrote prediction-focused 3D explorer HTML to {output_path}")


if __name__ == "__main__":
    main()
