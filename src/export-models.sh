#!/bin/bash  

### bash export-models.sh 2>&1 | tee export-models-log.log
### bash export-models.sh ov-models-test 2>&1 | tee export-models-log.log

# Set default OUTDIR value if not provided as first argument
OUT_DIR_ROOT=${1:-"ov-models"}
mkdir -p $OUT_DIR_ROOT

# Sample configuration: one model, int4 weights, ASYM with 128 and SYM with -1 
MODEL_IDS=(  
    "meta-llama/Llama-3.2-1B-Instruct"  
)  
WEIGHT_FORMATS=("int4")  
ASYM_GROUP_SIZES=(128)  
SYM_GROUP_SIZES=(-1)  

# # Sample configuration: three models, int4 weights, ASYM with 16,32,128 and SYM with -1,16,32,128
# MODEL_IDS=(  
#     "meta-llama/Llama-3.2-1B-Instruct"  
#     "Qwen/Qwen3-8B"  
#     "Qwen/Qwen2.5-1.5B-Instruct"  
# )  
  
# WEIGHT_FORMATS=("int4")  
# ASYM_GROUP_SIZES=(16 32 128)  
# SYM_GROUP_SIZES=(-1 16 32 128)  
  
echo "Starting batch export of HF models to OpenVINO..."  
echo "Models: ${MODEL_IDS[@]}"  
echo "ASYM group sizes: ${ASYM_GROUP_SIZES[@]}"  
echo "SYM group sizes: ${SYM_GROUP_SIZES[@]}"  
echo "Weight formats: ${WEIGHT_FORMATS[@]}"  
echo ""  


export_model() {  
    local MODEL_ID="$1"  
    local GROUP_SIZE="$2"  
    local WEIGHT_FORMAT="$3"  
    local SYM_FLAG="$4"  
    local SYM_LABEL="$5"  
      
    MODEL_NAME=$(basename "$MODEL_ID")
    OUTPUT_DIR="$OUT_DIR_ROOT/${MODEL_NAME}#${WEIGHT_FORMAT}#${SYM_LABEL}#g_${GROUP_SIZE}#ov"

    echo "----------------------------------------"
    echo "Exporting model: $MODEL_ID"
    echo "Weight format: $WEIGHT_FORMAT ; Group size: $GROUP_SIZE ; Symmetry: $SYM_LABEL"
    echo "Output directory: $OUTPUT_DIR"
    echo "----------------------------------------"
      
    # Export command with conditional --sym flag  
    if [ -n "$SYM_FLAG" ]; then  
        optimum-cli export openvino \
            --model "$MODEL_ID" \
            --weight-format "$WEIGHT_FORMAT" \
            --group-size "$GROUP_SIZE" \
            --sym \
            "$OUTPUT_DIR"
    else  
        optimum-cli export openvino \
            --model "$MODEL_ID" \
            --weight-format "$WEIGHT_FORMAT" \
            --group-size "$GROUP_SIZE" \
            "$OUTPUT_DIR"
    fi  
      
    echo "Export completed for $MODEL_ID ; $WEIGHT_FORMAT ; $SYM_LABEL ; $GROUP_SIZE"  
    echo ""  
}  
  
# Loop through all combinations  
for MODEL_ID in "${MODEL_IDS[@]}"; do  
    for WEIGHT_FORMAT in "${WEIGHT_FORMATS[@]}"; do  
        # ASYM combinations
        for GROUP_SIZE in "${ASYM_GROUP_SIZES[@]}"; do  
            export_model "$MODEL_ID" "$GROUP_SIZE" "$WEIGHT_FORMAT" "" "asym"  
        done  
          
        # SYM combinations
        for GROUP_SIZE in "${SYM_GROUP_SIZES[@]}"; do  
            export_model "$MODEL_ID" "$GROUP_SIZE" "$WEIGHT_FORMAT" "--sym" "sym"  
        done  
    done  
done  

echo "All exports completed successfully!"