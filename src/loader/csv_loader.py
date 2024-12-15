import polars as pl


def load_multi_header_csv(path: str, num_rows: int) -> pl.DataFrame:
    # まずはヘッダー部分を読み込む
    header = pl.read_csv(path, n_rows=num_rows, has_header=False)

    # ヘッダーを結合して新しい列名を作成
    combined_header = [
        "_".join(map(str, filter(None, col))) for col in zip(*header.to_numpy())
    ]

    # データ部分を読み込む
    df = pl.read_csv(path, skip_rows=num_rows, has_header=False)

    # 列名を設定
    df.columns = combined_header

    return df
