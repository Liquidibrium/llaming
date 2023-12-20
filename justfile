#!/usr/bin/env just --justfile

init-python:
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install poetry
    poetry install --no-root

download:
  python download.py

release:
  cargo build --release    

lint:
  cargo clippy

bin:
  cargo run --bin bin -- arg1

example:
  cargo run --example exname -- arg1