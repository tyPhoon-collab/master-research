def test_nearest_multiple():
    from tool.functions import nearest_multiple

    assert nearest_multiple(1, 2) == 2
    assert nearest_multiple(2, 2) == 2
    assert nearest_multiple(3, 2) == 4
    assert nearest_multiple(4, 2) == 4

    assert nearest_multiple(22050 * 30 // 256, 16) == 2576
