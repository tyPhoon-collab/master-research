def plot_spectrogram(data):
    import plotly.express as px

    assert data.ndim == 2, "data must be 2D array"

    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")
    return fig


def plot_waveform(data):
    import plotly.express as px

    assert data.ndim == 1, "data must be 1D array"

    fig = px.line(data)
    return fig


# TODO: consider deleting this function
# neptuneとの統合の簡易化のためにあるが、plotlyで代用できるのであれば削除する
def plot_mel_spectrogram_by_librosa(data):
    import librosa.display
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        data,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        sr=22050,
        hop_length=256,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig
