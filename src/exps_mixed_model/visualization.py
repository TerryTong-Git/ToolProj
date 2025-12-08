from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def surface_plot(
    grid: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    output_html: Optional[Path] = None,
    output_png: Optional[Path] = None,
    title: str = "Predicted task performance surface",
) -> go.Figure:
    """Create a 3D surface plot from a prediction grid."""
    required = {feature_x, feature_y, "prediction"}
    if not required.issubset(set(grid.columns)):
        missing = required.difference(grid.columns)
        raise ValueError(f"Grid missing required columns: {missing}")

    x_vals = np.sort(grid[feature_x].unique())
    y_vals = np.sort(grid[feature_y].unique())
    z_matrix = grid.pivot(index=feature_y, columns=feature_x, values="prediction").loc[y_vals, x_vals].values

    fig = go.Figure(data=[go.Surface(x=x_vals, y=y_vals, z=z_matrix, colorscale="Viridis")])
    fig.update_layout(scene=dict(xaxis_title=feature_x, yaxis_title=feature_y, zaxis_title="Predicted performance"), title=title)

    if output_html:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))

    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_png))
        except Exception:  # kaleido optional
            # PNG export is best-effort; skip if kaleido is unavailable.
            print("Skipping PNG export; install 'kaleido' to enable static image writing.")

    return fig
