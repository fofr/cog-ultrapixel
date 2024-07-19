# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
import subprocess
import time
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.download("ultrapixel_bundle", "ComfyUI/models/ultrapixel/")
        self.download("ultrapixel_huggingface_bundle", "/root/.cache/huggingface/hub/")

    def download(self, weight_str, dest):
        print(f"⏳ Downloading {weight_str} to {dest}")
        url = "https://weights.replicate.delivery/default/comfy-ui/ultrapixel"

        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", f"{url}/{weight_str}.tar", dest],
            close_fds=False,
        )
        elapsed_time = time.time() - start
        print(f"✅ Downloaded {weight_str} in {elapsed_time:.2f} seconds")

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        process = workflow["14"]["inputs"]
        process["prompt"] = kwargs["prompt"]
        process["seed"] = kwargs["seed"]
        process["width"] = kwargs["width"]
        process["height"] = kwargs["height"]
        process["stage_b_steps"] = kwargs["stage_b_steps"]
        process["stage_b_cfg"] = kwargs["stage_b_cfg"]
        process["stage_c_steps"] = kwargs["stage_c_steps"]
        process["stage_c_cfg"] = kwargs["stage_c_cfg"]
        process["controlnet_weight"] = kwargs["controlnet_weight"]

        if kwargs["image_filename"]:
            load_image = workflow["20"]["inputs"]
            load_image["image"] = kwargs["image_filename"]
        else:
            del workflow["20"]
            del process["controlnet_image"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        width: int = Input(
            default=2048,
            ge=1024,
            le=5120,
        ),
        height: int = Input(
            default=2048,
            ge=1024,
            le=5120,
        ),
        stage_b_steps: int = Input(
            default=10,
            ge=1,
            le=50,
        ),
        stage_b_cfg: float = Input(
            default=1.1,
            ge=0.1,
            le=10,
        ),
        stage_c_steps: int = Input(
            default=20,
            ge=1,
            le=50,
        ),
        stage_c_cfg: float = Input(
            default=4,
            ge=0.1,
            le=10,
        ),
        canny_control_image: Path = Input(
            description="Optional control image for canny controlnet",
            default=None,
        ),
        controlnet_weight: float = Input(
            default=0.7,
            ge=0.1,
            le=1.0,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        image_filename = None
        if canny_control_image:
            image_filename = self.filename_with_extension(canny_control_image, "canny_control_image")
            self.handle_input_file(canny_control_image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            image_filename=image_filename,
            seed=seed,
            width=width,
            height=height,
            stage_b_steps=stage_b_steps,
            stage_b_cfg=stage_b_cfg,
            stage_c_steps=stage_c_steps,
            stage_c_cfg=stage_c_cfg,
            controlnet_weight=controlnet_weight,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
