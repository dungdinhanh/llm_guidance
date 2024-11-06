#!/bin/bash



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
cmd="accelerate launch --num_processes 4 vlm_sd.py  $MODEL_FLAGS  $SAMPLE_FLAGS \
 --sample-dir runs/sd_test/"
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