#!/bin/bash

# Output CSV file
OUTPUT_CSV="overlap_benchmark_results.csv"
echo "size,overlap,prefill_time,decode_time,hit_rate,tokens_per_sec" > $OUTPUT_CSV

# Model path (use default if not provided)
MODEL="mistralai/Mixtral-8x7B-v0.1"

# Sizes to test
SIZES=(16 24 32)

echo "Starting overlap vs no-overlap benchmarks..."
echo "Results will be saved to $OUTPUT_CSV"

for size in "${SIZES[@]}"; do
    for overlap in 0 1; do
        if [ $overlap -eq 1 ]; then
            OVERLAP_LABEL="overlap"
            OVERLAP_FLAG="--overlap"
        else
            OVERLAP_LABEL="no-overlap"
            OVERLAP_FLAG=""
        fi

        echo "------------------------------------------------"
        echo "Testing Size: $size, Mode: $OVERLAP_LABEL"
        echo "------------------------------------------------"

        # Run latency.py and capture output
        # We use --cpu-offload 1 since overlap only applies to offloading
        OUTPUT=$(python latency.py \
            --model "$MODEL" \
            --cpu-offload 1 \
            $OVERLAP_FLAG \
            --input-token "$size" \
            --output-token "$size" \
            2>&1)

        # Extract values using grep and sed
        # Expected line format: prefill_time: 0.1234, decode_time: 0.5678, hit_rate: 0.9
        METRICS_LINE=$(echo "$OUTPUT" | grep "prefill_time:" | tail -n 1)

        if [ -z "$METRICS_LINE" ]; then
            echo "Error: Could not find metrics in output for size $size, overlap $overlap"
            echo "Output was:"
            echo "$OUTPUT"
            continue
        fi

        PREFILL=$(echo "$METRICS_LINE" | sed -E 's/.*prefill_time: ([0-9.]+).*/\1/')
        DECODE=$(echo "$METRICS_LINE" | sed -E 's/.*decode_time: ([0-9.]+).*/\1/')
        HIT_RATE=$(echo "$METRICS_LINE" | sed -E 's/.*hit_rate: ([0-9.]+).*/\1/')

        # Calculate tokens per second (output_tokens / (prefill + decode))
        # Use awk for floating point math
        TOKENS_PER_SEC=$(awk "BEGIN {print $size / ($PREFILL + $DECODE)}")

        echo "$size,$overlap,$PREFILL,$DECODE,$HIT_RATE,$TOKENS_PER_SEC" >> $OUTPUT_CSV

        echo "Size: $size, Overlap: $overlap"
        echo "  Prefill Time: $PREFILL s"
        echo "  Decode Time:  $DECODE s"
        echo "  Hit Rate:     $HIT_RATE"
        echo "  Tokens/s:     $TOKENS_PER_SEC"
    done
done

echo "------------------------------------------------"
echo "Benchmark complete. Results saved to $OUTPUT_CSV"
echo "------------------------------------------------"
cat $OUTPUT_CSV
