from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from depth_vbll import predict_checkpoint_on_dataframe
from tpc_html_common import (
    FIXED_TPC_R_MAX,
    FIXED_TPC_Z_MAX,
    cross_section_bounds,
    find_repo_root,
    group_to_components,
    load_plot_dataframe,
    output_path_or_default,
    require_plotly,
)


def _aggregate_heat(df, column: str, r_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    r_idx = np.digitize(df["s_r"], r_edges) - 1
    z_idx = np.digitize(df["sz_centered"], z_edges) - 1
    heat = np.full((len(z_edges) - 1, len(r_edges) - 1), np.nan)
    for iz in range(len(z_edges) - 1):
        for ir in range(len(r_edges) - 1):
            mask = (r_idx == ir) & (z_idx == iz)
            if np.any(mask):
                heat[iz, ir] = float(np.mean(df.loc[mask, column]))
    return heat


def _aggregate_quantile_heat(df, column: str, r_edges: np.ndarray, z_edges: np.ndarray, q: float) -> np.ndarray:
    r_idx = np.digitize(df["s_r"], r_edges) - 1
    z_idx = np.digitize(df["sz_centered"], z_edges) - 1
    heat = np.full((len(z_edges) - 1, len(r_edges) - 1), np.nan)
    for iz in range(len(z_edges) - 1):
        for ir in range(len(r_edges) - 1):
            mask = (r_idx == ir) & (z_idx == iz)
            if np.any(mask):
                heat[iz, ir] = float(np.quantile(df.loc[mask, column], q))
    return heat


def _plotting_edges(df) -> tuple[np.ndarray, np.ndarray]:
    r_upper = max(float(FIXED_TPC_R_MAX), float(np.nanmax(df["s_r"].to_numpy(dtype=float))))
    z_abs_max = max(float(FIXED_TPC_Z_MAX), float(np.nanmax(np.abs(df["sz_centered"].to_numpy(dtype=float)))))
    r_edges = np.linspace(0.0, r_upper * 1.02, 45)
    z_edges = np.linspace(-z_abs_max * 1.02, z_abs_max * 1.02, 45)
    return r_edges, z_edges


def build_figure(df, component_groups):
    go, make_subplots = require_plotly()
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Mean True d_center",
            "Mean Predicted d_center",
            "Mean Predicted Std",
            "Mean Absolute Error",
            "10th Percentile Predicted d_center",
            "Conservative Danger Score",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )

    r_edges, z_edges = _plotting_edges(df)
    centers_x = 0.5 * (r_edges[:-1] + r_edges[1:])
    centers_y = 0.5 * (z_edges[:-1] + z_edges[1:])
    bounds = cross_section_bounds()

    df = df.copy()
    df["pred_danger_score"] = df["pred_mean"] - df["pred_std"]

    panel_specs = [
        ("mean", "d_center", "Viridis_r", "mean true d_center", 1, 1, 0.29, 0.79),
        ("mean", "pred_mean", "Viridis_r", "mean predicted d_center", 1, 2, 0.64, 0.79),
        ("mean", "pred_std", "Magma", "mean predicted std", 1, 3, 0.99, 0.79),
        ("mean", "pred_abs_error", "Plasma", "mean absolute error", 2, 1, 0.29, 0.21),
        ("quantile", "pred_mean", "Viridis_r", "p10 predicted d_center", 2, 2, 0.64, 0.21),
        ("mean", "pred_danger_score", "Cividis_r", "mean(pred_mean - pred_std)", 2, 3, 0.99, 0.21),
    ]

    filter_map = group_to_components(component_groups)
    for component in sorted(component_groups):
        filter_map[component] = [component]
    filter_order = ["all", "top", "bottom", "side", *sorted(component_groups)]
    filter_labels = {
        "all": "All Components",
        "top": "Top Only",
        "bottom": "Bottom Only",
        "side": "Side Only",
        **{component: component for component in sorted(component_groups)},
    }

    heat_trace_indices: dict[str, list[int]] = {key: [] for key in filter_order}
    for filter_key in filter_order:
        allowed = set(filter_map[filter_key])
        filtered_df = df[df["source_component"].isin(allowed)].copy()
        for panel_idx, (agg_mode, column, colorscale, cbar_title, row, col, cbar_x, cbar_y) in enumerate(panel_specs):
            if agg_mode == "quantile":
                heat = _aggregate_quantile_heat(filtered_df, column, r_edges, z_edges, q=0.10)
            else:
                heat = _aggregate_heat(filtered_df, column, r_edges, z_edges)
            fig.add_trace(
                go.Heatmap(
                    x=centers_x,
                    y=centers_y,
                    z=heat,
                    colorscale=colorscale,
                    visible=(filter_key == "all"),
                    colorbar={
                        "title": cbar_title,
                        "len": 0.26,
                        "y": cbar_y,
                        "yanchor": "middle",
                        "thickness": 10,
                        "x": cbar_x,
                        "xanchor": "left",
                    },
                    hovertemplate=f"s_r=%{{x:.1f}}<br>sz_centered=%{{y:.1f}}<br>{cbar_title}=%{{z:.2f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            heat_trace_indices[filter_key].append(len(fig.data) - 1)

    for _, _, _, _, row, col, _, _ in panel_specs:
        fig.add_trace(
            go.Scatter(
                x=bounds["x"],
                y=bounds["y"],
                mode="lines",
                line={"color": "black", "dash": "dash", "width": 2},
                name="TPC bounds",
                showlegend=(row == 1 and col == 1),
                legendgroup="TPC bounds",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.update_xaxes(title_text="s_r", row=row, col=col)
            fig.update_yaxes(title_text="sz_centered", row=row, col=col)

    filter_buttons = []
    all_heat_indices = [idx for indices in heat_trace_indices.values() for idx in indices]
    for filter_key in filter_order:
        visible = [False] * len(all_heat_indices)
        wanted = set(heat_trace_indices[filter_key])
        for pos, trace_index in enumerate(all_heat_indices):
            visible[pos] = trace_index in wanted
        filter_buttons.append(
            {
                "label": filter_labels[filter_key],
                "method": "restyle",
                "args": [{"visible": visible}, all_heat_indices],
            }
        )

    fig.update_layout(
        title="XLZD TPC 2D Prediction Explorer",
        template="plotly_white",
        height=980,
        width=2200,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": filter_buttons,
                "x": 0.0,
                "y": 1.10,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"l": 50, "r": 360, "t": 120, "b": 50},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prediction-focused 2D TPC explorer HTML.")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw CSV data directory. Defaults to <repo>/data.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--max-rows-per-component", type=int, default=25000, help="Rows to load per component before downsampling.")
    parser.add_argument("--max-points-per-component", type=int, default=3000, help="Points per component to plot.")
    parser.add_argument(
        "--prediction-checkpoint",
        type=str,
        default=None,
        help="Optional VBLL checkpoint path. Defaults to the global center-distance model checkpoint.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_path = output_path_or_default(args.output, "tpc_penetration_prediction_explorer_2d.html")
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
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        },
    )
    print(f"Wrote prediction-focused 2D explorer HTML to {output_path}")


if __name__ == "__main__":
    main()
