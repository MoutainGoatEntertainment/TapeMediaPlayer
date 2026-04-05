#!/usr/bin/env bash
cd "$(dirname "$0")"
uv run --with PySide6 --with sounddevice --with numpy --with av tape_player.py
