from cog import BasePredictor, Input, Path
import os
import time
import torch
from typing import List
from diffusers import (
    DDIMScheduler, 
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler
)

MODEL_NAME = "ThinkDiffusion/ThinkDiffusionXL"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
SDXL_MODEL_CACHE = "sdxl-cache/"
REFINER_MODEL_CACHE = "refiner-cache/"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=SDXL_MODEL_CACHE,
        )
        self.txt2img_pipe.to("cuda")
        t2 = time.time()
        print("Setup sdxl & refiner took: ", t2 - t1)


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DDIM",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        sdxl_kwargs = {}
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        pipe = self.txt2img_pipe

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }
        output = pipe(**common_args, **sdxl_kwargs)
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        output_paths = []
        for i, _ in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
