#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

URL="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
OUTPUT_ZIP="data/raw/bike_sharing.zip"

echo "Downloading dataset..."
wget -O "$OUTPUT_ZIP" "$URL"

echo "Unzipping data assets..."
unzip -o "$OUTPUT_ZIP" -d data/raw/

echo "Cleaning up workspace..."
rm "$OUTPUT_ZIP"
echo "Data ingestion complete."
