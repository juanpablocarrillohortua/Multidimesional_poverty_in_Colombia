import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# Unit cube: its 8 corners and the 12 triangles that close it.
CUBE_X = np.array([0, 0, 1, 1, 0, 0, 1, 1])
CUBE_Y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
CUBE_Z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
CUBE_FACES = {
    "i": [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    "j": [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    "k": [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
}


def discrete_plot_joint(
    df, x_col, y_col, weights=None, title=None, colorscale="Tealgrn"
):
    """Barras 3D de la distribución conjunta P(X, Y) de dos columnas
    discretas.

    Sin `weights`, cada fila cuenta una vez: son proporciones de la muestra.
    Con `weights` (por ejemplo, "PESO"), cada fila cuenta según su factor de
    expansión: son proporciones de la población, las mismas que da
    `pd.crosstab(..., values=df[weights], aggfunc="sum", normalize="all")`.
    """
    if weights is None:
        joint = pd.crosstab(df[x_col], df[y_col], normalize=True)
    else:
        # Una combinación sin filas queda en NaN al sumar los pesos
        joint = pd.crosstab(
            df[x_col],
            df[y_col],
            values=df[weights],
            aggfunc="sum",
            normalize=True,
        ).fillna(0)
    n_x, n_y = joint.shape
    n_max = max(n_x, n_y)
    x_label = x_col.replace("_", " ").title()
    y_label = y_col.replace("_", " ").title()
    width = 0.6
    default_title = f"Distribución conjunta de {x_label} y {y_label}"
    if weights is not None:
        default_title += " (ponderada)"

    fig = go.Figure()
    for i, x_value in enumerate(joint.index):
        for j, y_value in enumerate(joint.columns):
            p = joint.iat[i, j]
            fig.add_trace(
                go.Mesh3d(
                    x=i + CUBE_X * width,
                    y=j + CUBE_Y * width,
                    z=CUBE_Z * p,
                    **CUBE_FACES,
                    intensity=np.full(8, p),
                    coloraxis="coloraxis",
                    flatshading=True,
                    hovertemplate=(
                        f"{x_label}: {x_value}<br>{y_label}: {y_value}"
                        f"<br>P(X, Y): {p:.2%}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title or default_title,
        template="plotly_white",
        width=900,
        height=650,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        coloraxis={
            "colorscale": colorscale,
            "cmin": 0,
            "cmax": joint.to_numpy().max(),
            "colorbar": {"title": "P(X, Y)", "tickformat": ".0%", "len": 0.6},
        },
        scene={
            "xaxis": {
                "title": x_label,
                "tickvals": np.arange(n_x) + width / 2,
                "ticktext": joint.index,
                "tickfont": {"size": 11},
            },
            "yaxis": {
                "title": y_label,
                "tickvals": np.arange(n_y) + width / 2,
                "ticktext": joint.columns,
                "tickfont": {"size": 11},
            },
            "zaxis": {"title": "P(X, Y)", "tickformat": ".0%"},
            "aspectmode": "manual",
            "aspectratio": {"x": n_x / n_max, "y": n_y / n_max, "z": 0.6},
            # Look slightly below the centre so the labels are not cropped.
            "camera": {
                "eye": {"x": 1.3, "y": 1.3, "z": 0.9},
                "center": {"x": 0, "y": 0, "z": -0.2},
            },
        },
    )
    fig.show()


def continuous_plot_joint(
    df, x_col, y_col, title=None, colorscale="Tealgrn", grid_size=60
):
    """3D surface of the joint density f(x, y) of two continuous columns,
    with the marginal density of each one in its own 2D panel."""
    data = df[[x_col, y_col]].dropna()
    x_label = x_col.replace("_", " ").title()
    y_label = y_col.replace("_", " ").title()

    x = np.linspace(data[x_col].min(), data[x_col].max(), grid_size)
    y = np.linspace(data[y_col].min(), data[y_col].max(), grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    kde = gaussian_kde(data.to_numpy().T)
    joint = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    # Same bandwidth factor as the joint estimate, so these are exactly its
    # marginals (the surface integrated over the other variable).
    marginal_x = gaussian_kde(data[x_col], bw_method=kde.factor)(x)
    marginal_y = gaussian_kde(data[y_col], bw_method=kde.factor)(y)

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "Densidad conjunta",
            f"Marginal de {x_label}",
            f"Marginal de {y_label}",
        ),
    )
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=joint.reshape(grid_x.shape),
            colorscale=colorscale,
            showscale=False,
            hovertemplate=(
                f"{x_label}: %{{x}}<br>{y_label}: %{{y}}"
                "<br>f(x, y): %{z:.3g}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    line = {"color": sample_colorscale(colorscale, 0.9)[0]}
    fig.add_trace(
        go.Scatter(x=x, y=marginal_x, fill="tozeroy", line=line), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=y, y=marginal_y, fill="tozeroy", line=line), row=2, col=2
    )

    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text="f(x)", exponentformat="e", row=1, col=2)
    fig.update_xaxes(title_text=y_label, row=2, col=2)
    fig.update_yaxes(title_text="f(y)", exponentformat="e", row=2, col=2)
    fig.update_layout(
        title=title or f"Distribución conjunta de {x_label} y {y_label}",
        template="plotly_white",
        width=1100,
        height=600,
        showlegend=False,
        margin={"l": 0, "r": 30, "t": 90, "b": 30},
        scene={
            "xaxis": {"title": x_label},
            "yaxis": {"title": y_label},
            "zaxis": {"title": "f(x, y)", "exponentformat": "e"},
            # Pulled back so the axis titles are not cropped.
            "camera": {
                "eye": {"x": 1.7, "y": 1.7, "z": 1.0},
                "center": {"x": 0, "y": 0, "z": -0.15},
            },
        },
    )
    fig.show()
