#!/usr/bin/env bash
# Development script: Compile TypeScript frontend and start web server
set -e

cd "$(dirname "$0")"

echo "Compiling TypeScript frontend..."
cd web
npx tsc
cd ..

echo "Starting PyRagix web server..."
uv run python -m web.server
