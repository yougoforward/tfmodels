#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
python "${WORK_DIR}"/model_test.py

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
# cd "${WORK_DIR}/${DATASET_DIR}"
# sh download_and_convert_voc2012.sh
# # Go back to original directory.
# cd "${CURRENT_DIR}"

# Set up the working directories.
PASCAL_FOLDER="pascal_voc_seg"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=30000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="trainval" \
  --model_variant="resnet_v1_50" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --train_batch_size=16 \
  --num_clones=4 \
  --base_learning_rate=0.007 \
  --weight_decay=0.0001 \
  --initialize_last_layer=false \
  --last_layer_gradient_multiplier=10.0 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/resnet_v1_50/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
# python "${WORK_DIR}"/eval.py \
#   --logtostderr \
#   --eval_split="val" \
#   --model_variant="resnet_v1_50_beta" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --eval_crop_size="513,513" \
#   --multi_grid=1 \
#   --multi_grid=2 \
#   --multi_grid=4 \
#   --base_learning_rate=0.007 \
#   --weight_decay=0.0001 \
#   --aspp_with_batch_norm=true \
#   --aspp_with_separable_conv=false \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --eval_logdir="${EVAL_LOGDIR}" \
#   --dataset_dir="${PASCAL_DATASET}" \
#   --max_number_of_evaluations=1










# # Visualize the results.
# python "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="val" \
#   --model_variant="resnet_v1_50_beta" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --vis_crop_size="513,513" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${PASCAL_DATASET}" \
#   --max_number_of_iterations=1

# # Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

# python "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --num_classes=21 \
#   --crop_size=513 \
#   --crop_size=513 \
#   --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
