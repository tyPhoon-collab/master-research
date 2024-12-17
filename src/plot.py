def plot_spectrogram(data):
    import plotly.express as px

    fig = px.imshow(data, color_continuous_scale="RdBu_r", origin="lower")

    return fig


# TODO: できれば無い方が良い
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
