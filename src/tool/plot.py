import io

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from PIL import Image
from PIL.ImageFile import ImageFile


def plot_spectrogram(data) -> go.Figure:
    assert data.ndim == 2, "data must be 2D array. data.ndim: {}".format(data.ndim)

    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")
    return fig


def plot_waveform(data) -> go.Figure:
    assert data.ndim == 1, "data must be 1D array. data.ndim: {}".format(data.ndim)

    fig = px.line(data)
    return fig


def plot_multiple(figs: list[go.Figure]) -> go.Figure:
    fig = sp.make_subplots(
        rows=len(figs), cols=1, subplot_titles=[f"Plot {i+1}" for i in range(len(figs))]
    )

    for i, f in enumerate(figs):
        for trace in f.data:
            fig.add_trace(trace, row=i + 1, col=1)

    fig.update_layout(height=200 * len(figs))
    return fig


def fig_to_pil_image(fig: go.Figure) -> ImageFile:
    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    return img
