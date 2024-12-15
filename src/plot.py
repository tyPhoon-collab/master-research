import plotly.express as px


def plot_spectrogram(data):
    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")

    return fig
