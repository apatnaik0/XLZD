from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tpc_html_common import (
    FIXED_TPC_R_MAX,
    FIXED_TPC_Z_MAX,
    component_color_map,
    cylinder_wireframe_xyz,
    find_repo_root,
    load_plot_dataframe,
    output_path_or_default,
    require_plotly,
    visibility_buttons,
)


def build_figure(df, component_groups):
    go, make_subplots = require_plotly()
    components = sorted(df["source_component"].unique())
    colors = component_color_map(components)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Initial Positions In 3D", "Final Positions In 3D"),
        horizontal_spacing=0.04,
    )

    trace_component_meta: list[str | None] = []

    for component in components:
        part = df[df["source_component"] == component]
        hover = np.column_stack(
            [
                part["source_component"],
                part["component_group"],
                np.round(part["d_center"], 2),
                np.round(part["r"], 2),
                np.round(part["z_from_center"], 2),
            ]
        )
        fig.add_trace(
            go.Scatter3d(
                x=part["sx"],
                y=part["sy"],
                z=part["sz_centered"],
                mode="markers",
                marker={"size": 3, "opacity": 0.5, "color": colors[component]},
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
                    "d_center=%{customdata[2]}<br>"
                    "r=%{customdata[3]}<br>"
                    "z_from_center=%{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        trace_component_meta.append(component)

        fig.add_trace(
            go.Scatter3d(
                x=part["x"],
                y=part["y"],
                z=part["z_centered_final"],
                mode="markers",
                marker={"size": 3, "opacity": 0.5, "color": colors[component]},
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
                    "d_center=%{customdata[2]}<br>"
                    "r=%{customdata[3]}<br>"
                    "z_from_center=%{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        trace_component_meta.append(component)

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

    buttons = visibility_buttons(trace_component_meta, component_groups)

    scene_common = {
        "xaxis_title": "x / sx",
        "yaxis_title": "y / sy",
        "zaxis_title": "z centered",
        "aspectmode": "cube",
        "xaxis": {"range": [-FIXED_TPC_R_MAX * 1.1, FIXED_TPC_R_MAX * 1.1]},
        "yaxis": {"range": [-FIXED_TPC_R_MAX * 1.1, FIXED_TPC_R_MAX * 1.1]},
        "zaxis": {"range": [-FIXED_TPC_Z_MAX * 1.1, FIXED_TPC_Z_MAX * 1.1]},
        "camera": {"eye": {"x": 1.65, "y": 1.45, "z": 1.1}},
    }
    fig.update_layout(
        title="XLZD TPC 3D Penetration Explorer",
        template="plotly_white",
        height=850,
        width=1700,
        legend={"groupclick": "togglegroup"},
        scene=scene_common,
        scene2=scene_common,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": buttons,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"l": 10, "r": 10, "t": 110, "b": 10},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 3D interactive TPC penetration explorer HTML.")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw CSV data directory. Defaults to <repo>/data.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--max-rows-per-component", type=int, default=25000, help="Rows to load per component before downsampling.")
    parser.add_argument("--max-points-per-component", type=int, default=2500, help="Points per component to plot.")
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_path = output_path_or_default(args.output, "tpc_penetration_explorer_3d.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, component_groups = load_plot_dataframe(
        data_dir,
        max_rows_per_component=args.max_rows_per_component,
        max_points_per_component=args.max_points_per_component,
    )
    fig = build_figure(df, component_groups)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Wrote 3D explorer HTML to {output_path}")


if __name__ == "__main__":
    main()
