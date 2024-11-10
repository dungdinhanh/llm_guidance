#!/bin/bash

#PBS -q gpuvolta
#PBS -P jp09
#PBS -l walltime=48:00:00
#PBS -l mem=256GB
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=256GB
#PBS -l wd
#PBS -l storage=scratch/jp09
#PBS -M adin6536@uni.sydney.edu.au
#PBS -o output_nci/run_vlm_dmd2_large.txt
#PBS -e error_nci/errors.txt


module load use.own
module load python3/3.9.2
module load sd22

MODEL_FLAGS=""

#SAMPLE_FLAGS="--per-proc-batch-size 1 --num-fid-samples 30000"
SAMPLE_FLAGS="--per-proc-batch-size 1  --num-fid-samples 30000 "
#SAMPLE_FLAGS="--batch_size 2 --num_samples 30000 --timestep_respacing 250"


cmd="cd ../"
#echo ${cmd}
#eval ${cmd}

base_folder="/scratch/jp09/dd9648/LVMguidance/"
hub_folder="/scratch/jp09/dd9648/hub/"
ref_folder="/scratch/jp09/dd9648/reference"
cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.5" "1.0" "2.0" )
scales=( "7.0"  )
seeds=("124" "125" "126" "127" "128" "129")

seeds=("130" "131" "132" "133" )
seeds=("134")
skips=("1")


#  --sample-dir runs/analyse/seed${seed}/lvm_sd_test/ --cfg-scale ${scale} --lvm-guidance --seed ${seed}"
# echo ${cmd}
# eval ${cmd}
# done
# done

for seed in "${seeds[@]}"
do
for scale in "${scales[@]}"
do
for skip in "${skips[@]}"
do
cmd="HUGGINGFACE_HUB_CACHE=${hub_folder} HFAI_DATASETS_DIR=/scratch/jp09/dd9648/data/ python3  vlm_dmd_shard_sk.py $MODEL_FLAGS  $SAMPLE_FLAGS --base_folder ${base_folder} \
 --sample-dir runs/exps/seed${seed}/largevlm_dmd2_skip${skip}/  --seed ${seed} --lvm-guidance --model_path liuhaotian/llava-v1.5-13b --skip ${skip}"
echo ${cmd}
eval ${cmd}
done
done
done

for seed in "${seeds[@]}"
do
  for scale in "${scales[@]}"
  do
  for skip in "${skips[@]}"
do
sample_file="${base_folder}runs/exps/seed${seed}/largevlm_dmd2_skip${skip}/reference/samples_30000x256x256x3.npz"

cmd="python3 evaluations/evaluator_tolog.py ${ref_folder}/VIRTUAL_MSCOCO_val_256x256_squ256.npz  \
 ${sample_file}"
echo ${cmd}
eval ${cmd}


cmd="python3 evaluations/evaluator_clipscore2.py --ref_path  ${sample_file}"
echo ${cmd}
eval ${cmd}
done
done
done

# for seed in "${seeds[@]}"
# do
# for scale in "${scales[@]}"
# do
# cmd="python analyse_vlm_sd_shard_negp_skip.py  $MODEL_FLAGS  $SAMPLE_FLAGS \
#  --sample-dir runs/analyse/seed${seed}/lvm_sd_test_negp_skip_wkwn/ --cfg-scale ${scale} --lvm-guidance --seed ${seed} --prompt_process 0"
# echo ${cmd}
# eval ${cmd}
# done
# done

# for seed in "${seeds[@]}"
# do
# for scale in "${scales[@]}"
# do
# cmd="python analyse_vlm_sd_shard_negp.py  $MODEL_FLAGS  $SAMPLE_FLAGS \
#  --sample-dir runs/analyse/seed${seed}/lvm_sd_test_negp_wdtn/ --cfg-scale ${scale} --lvm-guidance --seed ${seed} --prompt_process 1"
# echo ${cmd}
# eval ${cmd}
# done
# done

# for seed in "${seeds[@]}"
# do
# for scale in "${scales[@]}"
# do
# cmd="python analyse_vlm_sd_shard_posreplace.py  $MODEL_FLAGS  $SAMPLE_FLAGS \
#  --sample-dir runs/analyse/seed${seed}/lvm_sd_test_posreplace_rdtp/ --cfg-scale ${scale} --lvm-guidance --seed ${seed} "
# echo ${cmd}
# eval ${cmd}
# done
# done

# for seed in "${seeds[@]}"
# do
# for scale in "${scales[@]}"
# do
# cmd="python analyse_vlm_sd_shard.py  $MODEL_FLAGS  $SAMPLE_FLAGS \
#  --sample-dir runs/analyse/seed${seed}/lvm_sd_nolvm_test/ --cfg-scale ${scale} --seed ${seed}"
# echo ${cmd}
# eval ${cmd}
# done
# done


##
#for scale in "${scales[@]}"
#do
#cmd="python evaluations/evaluator_tolog.py reference/VIRTUAL_imagenet256_labeled.npz  \
# runs/DiT_clsfree/repg/IMN256/scale${scale}_cscale${cscale}/reference/samples_50000x256x256x3.npz"
#echo ${cmd}
#eval ${cmd}
#done







#cmd="python scripts_hfai_gdiff/classifier_sample.py --logdir runs/sampling/IMN64/conditional/ \
# ${MODEL_FLAGS} --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt \
# --classifier_depth 4 --model_path models/64x64_diffusion.pt ${SAMPLE_FLAGS}"
#echo ${cmd}
#eval ${cmd}