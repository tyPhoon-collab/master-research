def nearest_multiple(x: int, multiple: int) -> int:
    remainder = x % multiple
    if remainder < multiple // 2:
        return x - remainder
    else:
        return x + multiple - remainder


def fixed_time_axis_length(
    audio_duration: int,
    n_segments: int,
    sample_rate: int,
    hop_length: int,
) -> int:
    """
    モデルの入力は8や16の倍数である必要がある場合がある
    最も近い16の倍数になるように調整する
    """
    duration = audio_duration // n_segments
    frame_length = sample_rate * duration

    mel_estimated_length = frame_length // hop_length
    fixed_length = nearest_multiple(
        mel_estimated_length,
        multiple=16,
    )
    return fixed_length


def fixed_waveform_length(
    fixed_mel_length: int,
    hop_length: int,
) -> int:
    return fixed_mel_length * hop_length
