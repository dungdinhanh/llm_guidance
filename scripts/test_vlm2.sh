#!/bin/bash
export NCCL_P2P_DISABLE=1


MODEL_FLAGS=""

#SAMPLE_FLAGS="--per-proc-batch-size 1 --num-fid-samples 30000"
SAMPLE_FLAGS="--per-proc-batch-size 2  --num-fid-samples 4 --fix_seed"
#SAMPLE_FLAGS="--batch_size 2 --num_samples 30000 --timestep_respacing 250"


cmd="cd ../"
#echo ${cmd}
#eval ${cmd}

base_folder="./"

cmd="ls"
echo ${cmd}
eval ${cmd}

#scales=( "0.5" "1.0" "2.0" )
scales=( "1.5"  )


# This is classifier free guidance
#for scale in "${scales[@]}"
#do
#  for cscale in "${cscales[@]}"
#  do
for scale in "${scales[@]}"
do
cmd="WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 MARSV2_WHOLE_LIFE_STATE=0 python vlm_test3.py"
echo ${cmd}
eval ${cmd}
done
#done

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