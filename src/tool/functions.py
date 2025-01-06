def nearest_multiple(x: int, multiple: int) -> int:
    remainder = x % multiple
    if remainder < multiple // 2:
        return x - remainder
    else:
        return x + multiple - remainder
