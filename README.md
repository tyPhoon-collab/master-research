# 修士研究

[研究ログ](https://docs.craft.do/editor/d/65c9bdaf-9c31-c650-c37b-20d5fc61f516/AC95E3A3-50CE-457C-BACB-2C9D9E6AF1BC?s=ZddTeJy6ssL98nR8JktnuctSXsseedMoB9UoeW7ZedUV)
※ Web上でCraftが開きます

## セットアップ

### 計算機サーバーを用いる場合

Dockerをビルドし、bash上でuvを用いてセットアップする

Dockerのコンテナ名はMusic ControlNetの略である、mcnとしている

1. `docker compose -f ./.container/docker-compose.yaml up --build --detach`
2. `docker exec -it mcn bash`
3. `uv sync`

### 計算機サーバーを用いない場合

デバッグ時を考慮し、Macなどでも実行可能になっている

uvをインストールして、`uv sync`を実行する

## 実行例

```bash
# conf/config.yamlをもとに実行される
uv run main.py

# hydraの記法に従って、値を上書きすることができる
uv run main.py model=infer

# 上書き可能な値については、helpを参照
uv run main.py -h
```
