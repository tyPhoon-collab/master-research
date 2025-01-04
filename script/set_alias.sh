#!/bin/bash

function tu() {
    uv run main.py mode=train_unet "$@"
}

function tdw() {
    uv run main.py mode=train_diffwave "$@"
}

function doctor() {
    uv run main.py mode=doctor "$@"
}

function cln() {
    uv run main.py mode=clean "$@"
}