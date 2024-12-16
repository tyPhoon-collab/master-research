# Function to prompt the user and delete a folder
function Delete-Folder {
    param (
        [string]$FolderName
    )
    if (Test-Path $FolderName) {
        # Prompt the user
        $response = Read-Host "Do you want to delete the folder '$FolderName'? (y/n) [default: y]"

        # Treat Enter (empty response) as 'y'
        if (-not $response -or $response -match '^[yY]$') {
            Remove-Item -Recurse -Force $FolderName
            Write-Host "Deleted '$FolderName'."
        } else {
            Write-Host "Skipped '$FolderName'."
        }
    } else {
        Write-Host "Folder '$FolderName' does not exist. Skipping."
    }
}

# Check and delete specific folders
Delete-Folder ".neptune"
Delete-Folder "lightning_logs"
Delete-Folder "outputs"

Write-Host "Cleaning process completed!"