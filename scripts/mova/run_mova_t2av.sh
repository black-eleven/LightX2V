#!/bin/bash

# set path first
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MOVA-360p

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls mova \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/mova/mova.json \
--prompt "A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks." \
--negative_prompt "blurry, out of focus, low quality, muted audio, noisy audio, off-sync audio, artifacts" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_mova_t2av.mp4
