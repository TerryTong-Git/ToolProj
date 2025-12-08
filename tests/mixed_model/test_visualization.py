import pandas as pd

from src.exps_mixed_model import visualization


def _grid():
    rows = []
    for x in [0, 1]:
        for y in [0, 1]:
            rows.append({"x": x, "y": y, "prediction": float(x + y)})
    return pd.DataFrame(rows)


def test_surface_plot_generation():
    fig = visualization.surface_plot(_grid().rename(columns={"x": "a", "y": "b"}), feature_x="a", feature_y="b")
    assert fig.data


def test_all_feature_pairs_html_export(tmp_path):
    grid = _grid().rename(columns={"x": "feat1", "y": "feat2"})
    html_path = tmp_path / "plot.html"
    visualization.surface_plot(grid, feature_x="feat1", feature_y="feat2", output_html=html_path)
    assert html_path.exists()


def test_png_export_mocked(tmp_path, monkeypatch):
    grid = _grid().rename(columns={"x": "feat1", "y": "feat2"})
    png_path = tmp_path / "plot.png"

    captured = {}

    def fake_write_image(self, path):
        captured["path"] = path

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)
    visualization.surface_plot(grid, feature_x="feat1", feature_y="feat2", output_png=png_path)
    assert captured["path"].endswith("plot.png")


def test_colorscale_range():
    grid = _grid().rename(columns={"x": "feat1", "y": "feat2"})
    fig = visualization.surface_plot(grid, feature_x="feat1", feature_y="feat2")
    z_values = fig.data[0]["z"]
    assert z_values.min() == 0 and z_values.max() == 2
