from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile
    from plotly.graph_objs._figure import Figure


def plot_spectrogram(data) -> Figure:
    import plotly.express as px

    assert data.ndim == 2, "data must be 2D array. data.ndim: {}".format(data.ndim)

    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")
    return fig


def plot_waveform(data) -> Figure:
    import plotly.express as px

    assert data.ndim == 1, "data must be 1D array. data.ndim: {}".format(data.ndim)

    fig = px.line(data)
    return fig


def plot_multiple(figs: list[Figure]) -> Figure:
    import plotly.graph_objects as go
    import plotly.subplots as sp

    fig = sp.make_subplots(rows=len(figs), cols=1)

    for i, f in enumerate(figs):
        fig.add_trace(go.Figure(f), row=i + 1, col=1)

    return fig


def fig_to_pil_image(fig: Figure) -> ImageFile:
    import io

    from PIL import Image

    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    return img
