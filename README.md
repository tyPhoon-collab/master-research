# 修士研究

## プロジェクト

### ツール

- uv

### for Mac

開発目的でMacでも実行できるようにしてある

#### Setup

`script/darwin_setup.sh`を実行する。以下の処理が行われる

- CPUで動作するために、環境変数を設定する必要がある
  - `export PYTORCH_ENABLE_MPS_FALLBACK=1`
