#!/bin/bash

hub_folder="/hdd/dungda/hub/"
base_folder="./"
seeds=("134")
seeds=("134" "135" "136" "137" "138" )
# seeds=("137")

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
 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/dmd2_analyse_failurecases_comp1/case5/seed${seed}/fail/largelvm_dmd2_nolvm/ \
 --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder}"
 echo ${cmd}
 eval ${cmd}
 done

#  seeds=("140")

  for seed in "${seeds[@]}"
do
 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/dmd2_analyse_failurecases_comp1/case4/seed${seed}/fix/largelvm_dmd2_nolvm/ \
 --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder}"
#  echo ${cmd}
#  eval ${cmd}
 done

   for seed in "${seeds[@]}"
do
 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_dmd2_shard.py --sample-dir runs/dmd2_analyse_failurecases_simple1/case3/seed${seed}/fix_story/largelvm_dmd2_nolvm/ \
 --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder}"
#  echo ${cmd}
#  eval ${cmd}
 done