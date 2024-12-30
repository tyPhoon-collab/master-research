def test_plotly_fig_to_pil_image():
    import numpy as np

    from music_controlnet.plot import plot_waveform, plotly_fig_to_pil_image

    fig = plot_waveform(np.array([1, 2, 3, 4, 5]))
    img = plotly_fig_to_pil_image(fig)

    assert img is not None
