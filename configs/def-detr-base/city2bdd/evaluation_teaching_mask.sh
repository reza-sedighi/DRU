BATCH_SIZE=16
DATA_ROOT=./data
OUTPUT_DIR=./outputs/def-detr-base/city2bdd/teaching_mask/evaluation

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset bdd100k \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../city2bdd_teaching_mask.pth