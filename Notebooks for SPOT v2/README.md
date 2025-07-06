# Land use classification with TemporalViT and SPOT imagery

This repository contains three Jupyter notebooks for preprocessing, fine-tuning, and inferencing a Temporal Vision Transformer (TemporalViT) model using multi-temporal SPOT satellite imagery for land use classification in Khon Kaen, Thailand. The workflow leverages the Prithvi-100M pretrained model to classify six land use classes: Urban, Agricultural, Forest, Water, Oil Palm, and Para Rubber.

## Repository structure

- **Part 01: Preprocessing SPOT image and land use data** (`Part 01.ipynb`)
  - Preprocesses SPOT imagery (2016, 2020, 2022) and land use data.
  - Computes spectral indices (NDVI, NDWI), stacks multi-temporal bands, crops to a common extent, generates 112x112 patches, and splits data into training (70%), validation (10%), and testing (20%) sets.
  - Outputs: Stacked GeoTIFF (`stack.tif`) and patch pairs in `final_training_data`.

- **Part 02: Fine-tuning temporalViT** (`Part 02.ipynb`)
  - Fine-tunes the Prithvi-100M TemporalViT model using MMSegmentation for land use classification.
  - Configures the model with a Focal Loss function to handle class imbalance, trains on patch pairs, and visualizes sample predictions.
  - Outputs: Trained model checkpoints in `spot_a_training/exp01`.

- **Part 03: Inference with Fine-tuned model** (`Part 03.ipynb`)
  - Performs inference on new SPOT imagery using the fine-tuned model.
  - Visualizes input imagery and generates classified land use maps with a custom colormap.
  - Outputs: Predicted land use rasters in `output/inference_outputs`.
