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


def plotly_fig_to_pil_image(fig: Figure) -> ImageFile:
    import io

    from PIL import Image

    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    return img
