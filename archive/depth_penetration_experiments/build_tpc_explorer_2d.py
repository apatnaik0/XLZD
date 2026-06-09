from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tpc_html_common import (
    FIXED_TPC_R_MAX,
    FIXED_TPC_Z_MAX,
    component_color_map,
    cross_section_bounds,
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
        rows=2,
        cols=2,
        subplot_titles=(
            "Initial Position Cross-Section",
            "Final Position Cross-Section",
            "Initial Position Density",
            "Mean Center-Distance by Initial Position",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )

    trace_component_meta: list[str | None] = []
    bounds = cross_section_bounds()

    # Scatter traces per component for initial/final cross-sections.
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
            go.Scattergl(
                x=part["s_r"],
                y=part["sz_centered"],
                mode="markers",
                marker={"size": 5, "opacity": 0.45, "color": colors[component]},
                name=component,
                legendgroup=component,
                showlegend=True,
                customdata=hover,
                hovertemplate=(
                    "component=%{customdata[0]}<br>"
                    "group=%{customdata[1]}<br>"
                    "s_r=%{x:.2f}<br>"
                    "sz_centered=%{y:.2f}<br>"
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
            go.Scattergl(
                x=part["r"],
                y=part["z_centered_final"],
                mode="markers",
                marker={"size": 5, "opacity": 0.45, "color": colors[component]},
                name=component,
                legendgroup=component,
                showlegend=False,
                customdata=hover,
                hovertemplate=(
                    "component=%{customdata[0]}<br>"
                    "group=%{customdata[1]}<br>"
                    "r=%{x:.2f}<br>"
                    "z_centered_final=%{y:.2f}<br>"
                    "d_center=%{customdata[2]}<br>"
                    "r=%{customdata[3]}<br>"
                    "z_from_center=%{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        trace_component_meta.append(component)

    # TPC rectangle outlines
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
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
        trace_component_meta.append(None)

    r_upper = max(float(FIXED_TPC_R_MAX), float(np.nanmax(df["s_r"].to_numpy(dtype=float))))
    z_abs_max = max(float(FIXED_TPC_Z_MAX), float(np.nanmax(np.abs(df["sz_centered"].to_numpy(dtype=float)))))

    # Density heatmap
    fig.add_trace(
        go.Histogram2d(
            x=df["s_r"],
            y=df["sz_centered"],
            colorscale="Blues",
            xbins={"start": 0.0, "end": r_upper * 1.02, "size": (r_upper * 1.02) / 50.0},
            ybins={"start": -z_abs_max * 1.02, "end": z_abs_max * 1.02, "size": (2.0 * z_abs_max * 1.02) / 50.0},
            showscale=True,
            colorbar={
                "title": "count",
                "len": 0.32,
                "y": 0.19,
                "yanchor": "middle",
                "thickness": 14,
                "x": 0.47,
                "xanchor": "left",
            },
            hovertemplate="s_r=%{x}<br>sz_centered=%{y}<br>count=%{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    trace_component_meta.append(None)

    # Mean center-distance heatmap over initial position bins.
    r_edges = np.linspace(0.0, r_upper * 1.02, 45)
    z_edges = np.linspace(-z_abs_max * 1.02, z_abs_max * 1.02, 45)
    r_idx = np.digitize(df["s_r"], r_edges) - 1
    z_idx = np.digitize(df["sz_centered"], z_edges) - 1
    centers_x = 0.5 * (r_edges[:-1] + r_edges[1:])
    centers_y = 0.5 * (z_edges[:-1] + z_edges[1:])
    heat = np.full((len(z_edges) - 1, len(r_edges) - 1), np.nan)
    for iz in range(len(z_edges) - 1):
        for ir in range(len(r_edges) - 1):
            mask = (r_idx == ir) & (z_idx == iz)
            if np.any(mask):
                heat[iz, ir] = float(np.mean(df.loc[mask, "d_center"]))
    fig.add_trace(
        go.Heatmap(
            x=centers_x,
            y=centers_y,
            z=heat,
            colorscale="Viridis_r",
            colorbar={
                "title": "mean d_center",
                "len": 0.32,
                "y": 0.19,
                "yanchor": "middle",
                "thickness": 12,
                "x": 1.01,
                "xanchor": "left",
            },
            hovertemplate="s_r=%{x:.1f}<br>sz_centered=%{y:.1f}<br>mean d_center=%{z:.2f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    trace_component_meta.append(None)

    fig.update_xaxes(title_text="s_r", row=1, col=1)
    fig.update_yaxes(title_text="sz_centered", row=1, col=1)
    fig.update_xaxes(title_text="r", row=1, col=2)
    fig.update_yaxes(title_text="z_centered_final", row=1, col=2)
    fig.update_xaxes(title_text="s_r", row=2, col=1)
    fig.update_yaxes(title_text="sz_centered", row=2, col=1)
    fig.update_xaxes(title_text="s_r", row=2, col=2)
    fig.update_yaxes(title_text="sz_centered", row=2, col=2)

    buttons = visibility_buttons(trace_component_meta, component_groups)
    fig.update_layout(
        title="XLZD TPC 2D Penetration Explorer",
        template="plotly_white",
        height=1050,
        width=1750,
        legend={
            "groupclick": "togglegroup",
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
            "tracegroupgap": 6,
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": buttons,
                "x": 0.0,
                "y": 1.15,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"l": 50, "r": 340, "t": 130, "b": 50},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D interactive TPC penetration explorer HTML.")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw CSV data directory. Defaults to <repo>/data.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--max-rows-per-component", type=int, default=25000, help="Rows to load per component before downsampling.")
    parser.add_argument("--max-points-per-component", type=int, default=3000, help="Points per component to plot.")
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_path = output_path_or_default(args.output, "tpc_penetration_explorer_2d.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, component_groups = load_plot_dataframe(
        data_dir,
        max_rows_per_component=args.max_rows_per_component,
        max_points_per_component=args.max_points_per_component,
    )
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
    print(f"Wrote 2D explorer HTML to {output_path}")


if __name__ == "__main__":
    main()
