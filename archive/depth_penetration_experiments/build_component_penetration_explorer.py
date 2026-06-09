from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from depth_vbll import predict_checkpoint_on_dataframe
from tpc_html_common import find_repo_root, load_plot_dataframe, output_path_or_default, require_plotly


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.astype(float))
    if len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _component_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    true_global_cutoff = float(np.quantile(df["d_center"].to_numpy(dtype=float), 0.10))
    pred_global_cutoff = float(np.quantile(df["pred_mean"].to_numpy(dtype=float), 0.10))
    rows = []
    for component, part in df.groupby("source_component", sort=True):
        true_vals = part["d_center"].to_numpy(dtype=float)
        pred_vals = part["pred_mean"].to_numpy(dtype=float)
        pred_std = part["pred_std"].to_numpy(dtype=float)
        danger = pred_vals - pred_std
        rows.append(
            {
                "source_component": component,
                "count": int(len(part)),
                "true_mean": float(np.mean(true_vals)),
                "pred_mean": float(np.mean(pred_vals)),
                "true_p10": float(np.quantile(true_vals, 0.10)),
                "pred_p10": float(np.quantile(pred_vals, 0.10)),
                "true_frac_below_threshold": float(np.mean(true_vals < threshold)),
                "pred_frac_below_threshold": float(np.mean(pred_vals < threshold)),
                "true_frac_in_global_deepest10": float(np.mean(true_vals <= true_global_cutoff)),
                "pred_frac_in_global_deepest10": float(np.mean(pred_vals <= pred_global_cutoff)),
                "pred_mean_std": float(np.mean(pred_std)),
                "pred_mean_danger": float(np.mean(danger)),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["pred_p10", "pred_mean"], ascending=[True, True]).reset_index(drop=True)


def _panel_legend_annotations(threshold: float) -> list[dict]:
    box_style = {
        "showarrow": False,
        "align": "left",
        "bordercolor": "rgba(0,0,0,0.18)",
        "borderwidth": 1,
        "bgcolor": "rgba(255,255,255,0.88)",
        "font": {"size": 11},
    }
    return [
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.46,
            "y": 0.975,
            "text": "<span style='color:#1f77b4'>■</span> True mean<br><span style='color:#ff7f0e'>■</span> Predicted mean",
            **box_style,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.985,
            "y": 0.975,
            "text": "<span style='color:#2ca02c'>■</span> True p10<br><span style='color:#d62728'>■</span> Predicted p10",
            **box_style,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.46,
            "y": 0.645,
            "text": (
                "<span style='color:#9467bd'>■</span> True tail fraction<br>"
                "<span style='color:#8c564b'>■</span> Predicted tail fraction"
            ),
            **box_style,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.985,
            "y": 0.645,
            "text": "<span style='color:#1f77b4'>■</span> True d_center<br><span style='color:#ff7f0e'>■</span> Predicted d_center",
            **box_style,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.46,
            "y": 0.315,
            "text": "<span style='color:#1f77b4'>■</span> True CDF<br><span style='color:#d62728'>■</span> Predicted CDF",
            **box_style,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.985,
            "y": 0.315,
            "text": "<span style='color:#000000'>— —</span> error = std<br>Marker color: predicted d_center",
            **box_style,
        },
    ]


def build_figure(df: pd.DataFrame, *, threshold: float):
    go, make_subplots = require_plotly()

    df = df.copy()
    df["pred_danger_score"] = df["pred_mean"] - df["pred_std"]
    summary = _component_summary(df, threshold)
    components = summary["source_component"].tolist()
    default_component = components[0]
    tail_levels = [0.05, 0.10, 0.20, 0.50]
    tail_labels = ["Top 5%", "Top 10%", "Top 20%", "Top 50%"]
    true_global_cutoffs = {
        q: float(np.quantile(df["d_center"].to_numpy(dtype=float), q)) for q in tail_levels
    }
    pred_global_cutoffs = {
        q: float(np.quantile(df["pred_mean"].to_numpy(dtype=float), q)) for q in tail_levels
    }

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Mean d_center by Component",
            "10th Percentile d_center by Component",
            f"{default_component}: Fraction In Global Deepest Tails",
            f"{default_component}: True vs Predicted d_center",
            f"{default_component}: Empirical CDF",
            f"{default_component}: Uncertainty vs Error",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    y_labels = components[::-1]
    plot_summary = summary.set_index("source_component").loc[y_labels].reset_index()

    top_traces = [
        go.Bar(
            x=plot_summary["true_mean"],
            y=plot_summary["source_component"],
            orientation="h",
            name="True mean",
            marker={"color": "#1f77b4"},
            hovertemplate="component=%{y}<br>true mean d_center=%{x:.2f}<extra></extra>",
        ),
        go.Bar(
            x=plot_summary["pred_mean"],
            y=plot_summary["source_component"],
            orientation="h",
            name="Predicted mean",
            marker={"color": "#ff7f0e"},
            hovertemplate="component=%{y}<br>predicted mean d_center=%{x:.2f}<extra></extra>",
        ),
        go.Bar(
            x=plot_summary["true_p10"],
            y=plot_summary["source_component"],
            orientation="h",
            name="True p10",
            marker={"color": "#2ca02c"},
            hovertemplate="component=%{y}<br>true p10 d_center=%{x:.2f}<extra></extra>",
        ),
        go.Bar(
            x=plot_summary["pred_p10"],
            y=plot_summary["source_component"],
            orientation="h",
            name="Predicted p10",
            marker={"color": "#d62728"},
            hovertemplate="component=%{y}<br>predicted p10 d_center=%{x:.2f}<extra></extra>",
        ),
    ]
    top_positions = [(1, 1), (1, 1), (1, 2), (1, 2)]
    for trace, (row, col) in zip(top_traces, top_positions, strict=True):
        fig.add_trace(trace, row=row, col=col)

    detail_trace_indices: dict[str, list[int]] = {component: [] for component in components}
    for component in components:
        part = df[df["source_component"] == component].copy()
        true_vals = part["d_center"].to_numpy(dtype=float)
        pred_vals = part["pred_mean"].to_numpy(dtype=float)
        pred_std = part["pred_std"].to_numpy(dtype=float)
        abs_error = part["pred_abs_error"].to_numpy(dtype=float)
        danger = part["pred_danger_score"].to_numpy(dtype=float)

        visible = component == default_component

        true_tail_fracs = [float(np.mean(true_vals <= true_global_cutoffs[q])) for q in tail_levels]
        pred_tail_fracs = [float(np.mean(pred_vals <= pred_global_cutoffs[q])) for q in tail_levels]

        fig.add_trace(
            go.Bar(
                x=tail_labels,
                y=true_tail_fracs,
                name="True tail fraction",
                marker={"color": "#9467bd"},
                visible=visible,
                hovertemplate="tail=%{x}<br>true fraction=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        fig.add_trace(
            go.Bar(
                x=tail_labels,
                y=pred_tail_fracs,
                name="Predicted tail fraction",
                marker={"color": "#8c564b"},
                visible=visible,
                hovertemplate="tail=%{x}<br>predicted fraction=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        fig.add_trace(
            go.Histogram(
                x=true_vals,
                nbinsx=50,
                histnorm="probability density",
                name="True d_center",
                marker={"color": "#1f77b4"},
                opacity=0.60,
                visible=visible,
                hovertemplate="true d_center=%{x:.2f}<br>density=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        fig.add_trace(
            go.Histogram(
                x=pred_vals,
                nbinsx=50,
                histnorm="probability density",
                name="Predicted d_center",
                marker={"color": "#ff7f0e"},
                opacity=0.60,
                visible=visible,
                hovertemplate="predicted d_center=%{x:.2f}<br>density=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        x_true, y_true = _ecdf(true_vals)
        fig.add_trace(
            go.Scatter(
                x=x_true,
                y=y_true,
                mode="lines",
                name="True CDF",
                line={"color": "#1f77b4", "width": 2},
                visible=visible,
                hovertemplate="true d_center=%{x:.2f}<br>CDF=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        x_pred, y_pred = _ecdf(pred_vals)
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="lines",
                name="Predicted CDF",
                line={"color": "#d62728", "width": 2},
                visible=visible,
                hovertemplate="predicted d_center=%{x:.2f}<br>CDF=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=pred_std,
                y=abs_error,
                mode="markers",
                name="Prediction quality",
                marker={
                    "color": pred_vals,
                    "colorscale": "Viridis_r",
                    "size": 6,
                    "opacity": 0.65,
                    "showscale": True,
                    "colorbar": {
                        "title": "predicted d_center",
                        "x": 0.995,
                        "xanchor": "left",
                        "y": 0.14,
                        "yanchor": "middle",
                        "len": 0.22,
                        "thickness": 12,
                    },
                },
                customdata=np.column_stack([true_vals, pred_vals, danger]),
                visible=visible,
                hovertemplate=(
                    "predicted std=%{x:.2f}<br>"
                    "absolute error=%{y:.2f}<br>"
                    "true d_center=%{customdata[0]:.2f}<br>"
                    "predicted d_center=%{customdata[1]:.2f}<br>"
                    "danger score=%{customdata[2]:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=[float(np.min(pred_std)), float(np.max(pred_std))],
                y=[float(np.min(pred_std)), float(np.max(pred_std))],
                mode="lines",
                line={"color": "black", "dash": "dash", "width": 1.5},
                name="error = std",
                visible=visible,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        detail_trace_indices[component].append(len(fig.data) - 1)

    fig.update_xaxes(title_text="d_center", row=2, col=2)
    fig.update_yaxes(title_text="density", row=2, col=2)
    fig.update_xaxes(title_text="d_center", row=3, col=1)
    fig.update_yaxes(title_text="CDF", row=3, col=1, range=[0.0, 1.0])
    fig.update_xaxes(title_text="predicted std", row=3, col=2)
    fig.update_yaxes(title_text="absolute error", row=3, col=2)

    fig.update_xaxes(title_text="d_center", row=1, col=1)
    fig.update_xaxes(title_text="d_center", row=1, col=2)
    fig.update_xaxes(title_text="global deepest tail", row=2, col=1)
    fig.update_yaxes(title_text="fraction", row=2, col=1, range=[0.0, 1.0])

    total_traces = len(fig.data)
    detail_trace_positions = [idx for indices in detail_trace_indices.values() for idx in indices]
    detail_buttons = []
    for component in components:
        visible = [True] * total_traces
        allowed = set(detail_trace_indices[component])
        for idx in detail_trace_positions:
            visible[idx] = idx in allowed
        detail_buttons.append(
            {
                "label": component,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "annotations[2].text": f"{component}: Fraction In Global Deepest Tails",
                        "annotations[3].text": f"{component}: True vs Predicted d_center",
                        "annotations[4].text": f"{component}: Empirical CDF",
                        "annotations[5].text": f"{component}: Uncertainty vs Error",
                    },
                ],
            }
        )
    for annotation in _panel_legend_annotations(threshold):
        fig.add_annotation(**annotation)

    fig.update_layout(
        title="XLZD Component Penetration Explorer",
        template="plotly_white",
        height=1350,
        width=1650,
        barmode="group",
        showlegend=False,
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": detail_buttons,
                "x": 0.0,
                "y": 1.14,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        margin={"l": 70, "r": 40, "t": 150, "b": 50},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a component-level XLZD penetration explorer HTML.")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw CSV data directory. Defaults to <repo>/data.")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--max-rows-per-component", type=int, default=25000, help="Rows to load per component before downsampling.")
    parser.add_argument("--max-points-per-component", type=int, default=5000, help="Points per component to plot.")
    parser.add_argument("--danger-threshold", type=float, default=400.0, help="d_center threshold used for deep-penetration ranking.")
    parser.add_argument(
        "--prediction-checkpoint",
        type=str,
        default=None,
        help="Optional VBLL checkpoint path. Defaults to the global center-distance model checkpoint.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    output_path = output_path_or_default(args.output, "component_penetration_explorer.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_checkpoint = repo_root / "depth_penetration_experiments" / "artifacts" / "global_center_distance" / "vbll_regressor.pt"
    checkpoint_path = Path(args.prediction_checkpoint) if args.prediction_checkpoint else default_checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Prediction checkpoint not found: {checkpoint_path}. "
            "Run the global penetration notebook first or pass --prediction-checkpoint."
        )

    df, _ = load_plot_dataframe(
        data_dir,
        max_rows_per_component=args.max_rows_per_component,
        max_points_per_component=args.max_points_per_component,
    )
    df = predict_checkpoint_on_dataframe(df, checkpoint_path=checkpoint_path)
    fig = build_figure(df, threshold=float(args.danger_threshold))
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        },
    )
    print(f"Wrote component penetration explorer HTML to {output_path}")


if __name__ == "__main__":
    main()
