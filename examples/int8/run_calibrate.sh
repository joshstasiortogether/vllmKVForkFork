#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
work_dir=./work_dir/
log_dir=./cali_log/

# Using wikitext2 which is a built-in dataset
dataset="wikitext2"
save_dir=${work_dir}${dataset}/pth/
[ ! -d ${save_dir} ] && mkdir -p ${save_dir}
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

log=${log_dir}llama2-7b-${dataset}.log
echo "dataset=${dataset}, save_dir=${save_dir}, log=${log}"

# Run calibration
python calibrate.py ~/models/LLaMA-2-7B/ \
        --calib_dataset ${dataset} \
        --work_dir ${save_dir} \
        --device cuda \
        --calib_samples 128 \
        --calib_seqlen 2048 2>&1 | tee ${log} 

# Export parameters
log=${log_dir}llama2-7b-${dataset}_json.log
save_dir_path=${work_dir}${dataset}/
python export_kv_params.py \
    --work_dir ${save_dir} \
    --kv_params_dir ${save_dir_path} \
    --quant_group 128 2>&1 | tee ${log}
