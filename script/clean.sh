#!/bin/bash

# フォルダを確認して削除する関数
delete_folder() {
    local folder_name=$1
    if [ -d "$folder_name" ]; then
        echo "Do you want to delete the folder '$folder_name'? (y/n) [default: y]: "
        read -r answer
        # 入力が空の場合は 'y' とみなす
        answer=${answer:-y}
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
            rm -rf "$folder_name"
            echo "Deleted '$folder_name'."
        else
            echo "Skipped '$folder_name'."
        fi
    else
        echo "Folder '$folder_name' does not exist. Skipping."
    fi
}

# 各フォルダを確認して削除
delete_folder ".neptune"
delete_folder "lightning_logs"
delete_folder "outputs"

echo "Cleaning process completed!"