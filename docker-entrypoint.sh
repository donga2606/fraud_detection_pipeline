#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    exec python -m src.cli --help
fi

if [ "$1" = "dashboard" ]; then
    shift
    exec streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501 "$@"
fi

if [ "$1" = "python" ] || [ "$1" = "streamlit" ] || [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

exec python -m src.cli "$@"
