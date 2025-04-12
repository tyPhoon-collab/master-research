def test_fig_to_pil_image():
    import numpy as np

    from visualize.plot import fig_to_pil_image, plot_waveform

    fig = plot_waveform(np.array([1, 2, 3, 4, 5]))
    img = fig_to_pil_image(fig)

    assert img is not None
