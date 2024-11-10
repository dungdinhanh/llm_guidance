#!/bin/bash

hub_folder="/hdd/dungda/hub/"
base_folder="./"
seeds=("134")
seeds=("134" "135" "136" "137" "138" "139" "140" "141" "142" "143")

# for seed in "${seeds[@]}"
# do
# cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/analyse_prompt2/seed${seed}/lvm_dmd2/ \
#  --model_path liuhaotian/llava-v1.5-7b --lvm-guidance --seed ${seed} --fix_seed --base_folder ${base_folder}"
#  echo ${cmd}
#  eval ${cmd}

#  cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/analyse_prompt2/seed${seed}/lvm_dmd2_nolvm/ \
#  --model_path liuhaotian/llava-v1.5-7b --seed ${seed} --fix_seed --base_folder ${base_folder}"
#  echo ${cmd}
#  eval ${cmd}
#  done

 for seed in "${seeds[@]}"
do
cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/largeanalyse_prompt2/seed${seed}/largelvm_dmd2/ \
 --model_path liuhaotian/llava-v1.5-13b --lvm-guidance --seed ${seed} --fix_seed --base_folder ${base_folder}"
 echo ${cmd}
 eval ${cmd}

 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/largeanalyse_prompt2/seed${seed}/largelvm_dmd2_nolvm/ \
 --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder}"
 echo ${cmd}
 eval ${cmd}
 done