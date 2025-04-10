MODELS_DIR="/home/elise/Documents/M1-BIM/S2/TMEs/SBAS/tme08/models/scop95"
SCOP_TEST="/home/elise/Documents/M1-BIM/S2/TMEs/SBAS/tme08/scop/scop/scopTestSeq.fasta"
OUTPUT_DIR="/home/elise/Documents/M1-BIM/S2/TMEs/SBAS/tme08/searchResults/scop95"

mkdir -p "$OUTPUT_DIR"

for model_file in "$MODELS_DIR"/*.hmm; do
    tab_name=$(basename "$model_file" .hmm).out
    echo "Running: hmmsearch "$tab_name $model_file $SCOP_TEST""
    hmmsearch --domtblout "$OUTPUT_DIR/$tab_name" -E 1 "$model_file" "$SCOP_TEST"
done

echo "HMM models have been searched successfully."
