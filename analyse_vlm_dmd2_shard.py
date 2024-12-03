import torch
from diffusers import StableDiffusionXLPipeline
from diffusion.lvm_diffusion_xl import StableDiffusionXLLVMPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from datasets.coco_helper2 import load_data_caption_hfai_one_process
from utils.llava_utils import *

import argparse

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from diffusion.lvm_utils import *
import numpy as np
import random


def main():
    args = create_argparser().parse_args()

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "tianweiy/DMD2"
    ckpt_name = "dmd2_sdxl_4step_lora_fp16.safetensors"
    # Load model.

    if args.fix_seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enabled=False
        torch.backends.cudnn.deterministic=True

    base_folder = args.base_folder
    # Create folder to save samples
    sample_dir = os.path.join(base_folder, args.sample_dir)
    sample_folder_dir = os.path.join(sample_dir, f"images")
    prompt_folder_dir = os.path.join(sample_dir, f"prompts")
    analysis_folder_dir = os.path.join(sample_dir, f"analysis")

    os.makedirs(sample_folder_dir, exist_ok=True)
    os.makedirs(prompt_folder_dir, exist_ok=True)
    os.makedirs(analysis_folder_dir, exist_ok=True)

    # scenario 1: simple text
    # caption_loader = load_data_caption_hfai_one_process(split="val", batch_size=1)
    # caption_loader = [["A plate of spaghetti with red sauce and broccoli."]] # seed 140 -> wrong dish
    # caption_loader = [["Spaghetti with red sauce and broccoli on a plate"]]
    # caption_loader = [["Two men who are standing in the grass near a soccer ball."]] # seed 134 -> fix: 
    # caption_loader = [["A soccer ball is near to two standing men in the grass"]] 
    # caption_loader = [["In a living room, a cat and a dog sleeping on the floor"]] # bad for 136 
    # # -> fix:  
    # # caption_loader = [["a cat and a dog sleeping on the floor in a living room"]] # -> quite oke / poorer: # 
    # caption_loader = [["A living room scene with a dog and a cat sleeping on the floor."]]
    # caption_loader = [["Two computer monitors are on the desk with a chair at the desk"]] # bad for 136 #=> quite oke 
    # caption_loader = [["A computer chair at the desk with two computer monitors."]]
    
    # scenario 2: complicated text
    # caption_loader = [["A bustling, futuristic cityscape at sunset, where skyscrapers blend into giant, ancient trees; digital billboards flicker with holographic koi fish swimming across the screens, and pedestrians wear a mix of cyberpunk clothing and traditional robes, all illuminated by a soft, glowing mist"]]
    # caption_loader = [["A cozy library with a roaring fireplace, an open book on a vintage table, a cat sleeping on a cushioned chair, and a globe in the corner."]]
    # caption_loader = [["A medieval blacksmith’s shop with a blazing forge, metal tools hanging on the wall, a half-finished sword on the anvil, and a bucket of water nearby."]]
    caption_loader = [["An explorer’s tent in the jungle with a canvas map pinned to a board, binoculars, a compass, field journals filled with sketches, scattered specimen jars, a backpack leaning against a cot, mosquito netting, a lantern, and boots caked in mud."]]
    

    caption_iter = iter(caption_loader)

    pipe = StableDiffusionXLLVMPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16", device_map="balanced")
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora(lora_scale=1.0)  # we might want to make the scale smaller for community models

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # set up lvm
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map="auto"
    )

    process_prompt_mode = RDTP

    prompt=next(caption_iter)

    # LCMScheduler's default timesteps are different from the one we used for training 
    images=pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0, timesteps=[999, 749, 499, 249],
                            lvm_tokenizer=tokenizer, lvm_model=model, lvm_image_processor=image_processor,  
                            do_lvm_guidance=args.lvm_guidance, process_prompt_mode=process_prompt_mode, 
                            analysis_folder = analysis_folder_dir, track_image_prompt=True, skip=1).images
    
    for i in range(len(images)):
        image_file = os.path.join(sample_folder_dir, f"final_image{i}.png")
        images[i].save(image_file)
        prompt_file = os.path.join(prompt_folder_dir, f"prompt{i}.txt")
        f = open(prompt_file, "w")
        f.write(prompt[i])
        f.close()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="runs/analyse/samples")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--lvm-guidance", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fix_seed", action="store_true")
    parser.add_argument("--base_folder", type=str, default="./")
    return parser


if __name__ == '__main__':
    main()