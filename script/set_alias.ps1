function tu {
    uv run main.py mode=train_unet @args
}

function tdw {
    uv run main.py mode=train_diffwave @args
}

function doctor {
    uv run main.py mode=doctor @args
}

function cln {
    uv run main.py mode=clean @args
}