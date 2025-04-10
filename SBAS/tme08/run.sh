#!/bin/bash

ALIGNMENT_DIR="/home/elise/Documents/M1-BIM/S2/TMEs/SBAS/tme08/scop/scop/scop95/aln"
OUTPUT_DIR="/home/elise/Documents/M1-BIM/S2/TMEs/SBAS/tme08/models/scop95"

mkdir -p "$OUTPUT_DIR"

for aln_file in "$ALIGNMENT_DIR"/*.sto; do
    model_name=$(basename "$aln_file" .sto).hmm
    model_path="$OUTPUT_DIR/$model_name"
    
    echo "Running: hmmbuild $model_path $aln_file"
    hmmbuild "$model_path" "$aln_file"
done

echo "HMM models have been built successfully."
