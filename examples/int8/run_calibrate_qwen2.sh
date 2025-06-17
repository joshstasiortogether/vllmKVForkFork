#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
work_dir=./work_dir/
log_dir=./cali_log/

# Create directories if they don't exist
[ ! -d ${work_dir} ] && mkdir -p ${work_dir}
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

# Set model path - replace with your actual model path
MODEL_PATH="Qwen/Qwen2-7B-Instruct"

# Run calibration
save_dir=${work_dir}/qwen2/pth/
[ ! -d ${save_dir} ] && mkdir -p ${save_dir}

log=${log_dir}qwen2-7b-calibrate.log
echo "Calibrating Qwen2-7B, save_dir=${save_dir}, log=${log}"

python calibrate.py ${MODEL_PATH} \
    --work_dir ${save_dir} \
    --device cuda \
    --calib_samples 128 \
    --calib_seqlen 2048 2>&1 | tee ${log}

# Export KV cache parameters
log=${log_dir}qwen2-7b-export.log
save_dir_path=${work_dir}/qwen2/

python export_kv_params.py \
    --work_dir ${save_dir} \
    --kv_params_dir ${save_dir_path} \
    --quant_group 64 2>&1 | tee ${log} 