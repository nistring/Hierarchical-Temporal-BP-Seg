#!/bin/bash

# Simple script to run visualize_annotations.py
PROJECT_ROOT="$(pwd)"

# Default arguments
INPUT_DIR="${1:-$PROJECT_ROOT/raw/anno_edited}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/SUIT/demo/visualized_annotations_edited}"
VIDEO_DIR="${3:-$PROJECT_ROOT/raw/videos}"

# Run the script
python3 "$PROJECT_ROOT/visualize_annotations.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --stats-only \
    "${@:3}"
