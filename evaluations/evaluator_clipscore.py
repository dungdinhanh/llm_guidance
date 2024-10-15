from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import numpy as np
import argparse
import os
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
print("go here?")

def calculate_clip_score_batch(images, prompts):
    # images_int = (images * 255).astype("uint8")
    images = torch.from_numpy(images).permute(0,3,2,1).cuda()
    print(images)
    clip_score_vl = clip_score_fn(images, prompts).detach().cpu().item()
    return round(float(clip_score_vl), 4)

def calculate_clip_score(ref_path, batch_size):
    obj = np.load(ref_path)
    images = obj["arr_0"]
    labels = obj["arr_1"]
    labels_converted = []
    for label in labels:
        labels_converted.append(str(label))
    n = images.shape[0]
    no_batch = int(n/batch_size)
    total_clip_score = 0.0
    if no_batch * batch_size < n:
        no_batch += 1
    for i in range(no_batch):
        start_idx = i * batch_size
        end_idx = i * batch_size + batch_size
        if end_idx > n:
            end_idx = n
        number_images = end_idx - start_idx
        batch_images = images[start_idx: end_idx]
        batch_labels = labels_converted[start_idx: end_idx]
        clip_score_batch = calculate_clip_score_batch(batch_images, batch_labels)
        total_clip_score += (clip_score_batch * number_images)
    final_clip_score = total_clip_score/float(n)

    return final_clip_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()

    print("here?")

    folder_ref = os.path.dirname(args.sample_batch)
    clip_score_file = os.path.join(folder_ref, "clip_score.csv")

    print("here2?")
    final_clip_score = calculate_clip_score(args.sample_batch, 1)

    print("here3")
    f = open(clip_score_file, "w")
    f.write("CLIP score,\n")
    f.write(f"{final_clip_score},\n")
    f.close()





