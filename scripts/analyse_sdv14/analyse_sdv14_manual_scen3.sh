#!/bin/bash

hub_folder="/hdd/dungda/hub/"
base_folder="./"
seeds=("134")
seeds=("134" "135" "136" "137" "138" "139" )
seeds=("135")
cfgs=("6.5" "7.5" "8.5" "9.5")

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
 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_sdv14_shard_fixmanual.py \
  --sample-dir runs/sdv14_analyse_manualfix_cases_simple1/case1/seed${seed}/fail/largelvm_dmd2_nolvm/ --lvm-guidance \
 --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder} --cfg_scale 7.5"
 echo ${cmd}
 eval ${cmd}

 done

  for seed in "${seeds[@]}"
do
for cfg in "${cfgs[@]}"
do
 cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/hdd/datasets/ python analyse_vlm_sdv14_shard.py \
  --sample-dir runs/sdv14_analyse_failurecases_simple1/case1/seed${seed}/fail_increasecfg/largelvm_dmd2_nolvm_cfg${cfg}/ \
    --model_path liuhaotian/llava-v1.5-13b --seed ${seed} --fix_seed --base_folder ${base_folder} --cfg_scale ${cfg}"
#  echo ${cmd}
#  eval ${cmd}
 done
 done