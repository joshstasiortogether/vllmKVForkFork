#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
datasets_path=/home/datasets/
work_dir=./work_dir/
# Using English evaluation datasets
datasets_name="mmlu"
log_dir=./cali_log/

for i in $datasets_name;
do
    calib_dataset_path=${datasets_path}${i}/
    save_dir=${work_dir}$i/pth/
    [ ! -d ${save_dir} ] && mkdir -p ${save_dir}
    [ ! -d ${log_dir} ] && mkdir -p ${log_dir}
    
    log=${log_dir}llama2-7b-datasets_$i.log
    echo "i=$i, calib_dataset_path=${calib_dataset_path}, save_dir=${save_dir}, log=${log}"
    
    # Using Llama-2-7b-hf model instead of Chinese model
    python calibrate.py meta-llama/Llama-2-7b-hf \
            --calib_dataset $i \
            --dataset_path ${calib_dataset_path} \
            --work_dir ${save_dir} \
            --device cuda \
            --calib_samples 128 \
            --calib_seqlen 2048 2>&1 | tee ${log} 
    
    log=${log_dir}llama2-7b-datasets_${i}_json.log
    save_dir_path=${work_dir}$i/
    python export_kv_params.py \
        --work_dir ${save_dir} \
        --kv_params_dir ${save_dir_path} \
        --quant_group 128 2>&1 | tee ${log} 
done

