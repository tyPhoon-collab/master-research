import os
import shutil

from rich.console import Console
from rich.prompt import Confirm

console = Console()


def delete_folder(folder_path):
    if not os.path.exists(folder_path):
        console.print(
            f"Folder '{folder_path}' does not exist. Skipping.", style="yellow"
        )
        return

    accept = Confirm.ask(f"Delete folder '{folder_path}'?", default=True)
    if not accept:
        console.print(f"Skipped '{folder_path}'.", style="yellow")
        return

    try:
        shutil.rmtree(folder_path)
        console.print(f"Deleted '{folder_path}'.", style="green")
    except OSError as e:
        console.print(f"Error deleting '{folder_path}': {e}", style="red")


def find_pycache_folders(parent_folder: str):
    return [
        os.path.join(root, "__pycache__")
        for root, dirs, _ in os.walk(parent_folder)
        if "__pycache__" in dirs
    ]


def clean():
    folders_to_delete = [
        ".neptune",
        "lightning_logs",
        "outputs",
        "src/master.egg-info",
        *find_pycache_folders("src"),
        *find_pycache_folders("tests"),
    ]

    console.print("Starting cleaning process...")
    for folder in folders_to_delete:
        delete_folder(folder)

    console.print("Cleaning process completed!", style="bold green")


if __name__ == "__main__":
    clean()
