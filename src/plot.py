import plotly.express as px
from plotly.graph_objs._figure import Figure


def plot_spectrogram(data) -> Figure:
    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")

    return fig
