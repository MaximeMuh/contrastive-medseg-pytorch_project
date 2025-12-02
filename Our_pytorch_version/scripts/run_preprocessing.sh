# Small helper script to run the ACDC preprocessing pipeline.
# We call the Python script that does: N4 bias correction, normalization,
# resampling and cropping of the ACDC dataset.

# Configuration (change these paths if your data is elsewhere)
INPUT_DIR="database/acdc"
OUTPUT_BIAS_DIR="database/acdc_bias_corr"
OUTPUT_CROPPED_DIR="database/acdc_bias_corr_cropped"

echo "======================================"
echo "ACDC Dataset Preprocessing"
echo "======================================"
echo ""
echo "Input directory: $INPUT_DIR"
echo "Bias-corrected output: $OUTPUT_BIAS_DIR"
echo "Preprocessed output: $OUTPUT_CROPPED_DIR"
echo ""


if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Run the Python preprocessing script
python3 pytorch_version/scripts/preprocess_acdc.py \
    --input_dir "$INPUT_DIR" \
    --output_bias_dir "$OUTPUT_BIAS_DIR" \
    --output_cropped_dir "$OUTPUT_CROPPED_DIR"

echo ""
echo "======================================"
echo "Preprocessing complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Update configs/config_acdc.yaml with these paths:"
echo "   data_path_tr: $OUTPUT_BIAS_DIR/patient"
echo "   data_path_tr_cropped: $OUTPUT_CROPPED_DIR/patient"
echo ""
echo "2. Run the training pipeline (for example):"
echo "   jupyter notebook Notebook_Full_Pipeline.ipynb"