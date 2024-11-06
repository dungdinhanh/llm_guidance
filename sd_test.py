import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    pipe.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    result = pipe(prompt).images[0]
    result.save(f"result_{rank}.png")



if __name__ == "__main__":
    # main(0)
    ngpus = torch.cuda.device_count()
    run_inference(0, 3)
    # torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)