import torch
from torch import autocast
from download import load_model_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import time
from tqdm import tqdm, trange
from txt2img import chunk, numpy_to_pil, put_watermark, load_replacement, check_safety
from imwatermark import WatermarkEncoder
from PIL import Image
from einops import rearrange
import os
import numpy as np
from omegaconf import OmegaConf
import base64
from io import BytesIO
import json
from pytorch_lightning import seed_everything


CONFIG_FILE = "configs/stable-diffusion/v1-inference.yaml"


def init():
    global model  # needed for bananna optimizations
    config = OmegaConf.load(CONFIG_FILE)

    model = load_model_from_config(config, "/models/model-epoch07-float16.ckpt")
    model.cuda()
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


def decodeBase64Image(imageStr: str) -> Image:
    return Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))


def truncateInputs(inputs: dict):
    clone = inputs.copy()
    if "modelInputs" in clone:
        modelInputs = clone["modelInputs"] = clone["modelInputs"].copy()
        for item in ["init_image", "mask_image"]:
            if item in modelInputs:
                modelInputs[item] = modelInputs[item][0:6] + "..."
    return clone


def inference(all_inputs: dict) -> dict:
    global model

    print(json.dumps(truncateInputs(all_inputs), indent=2))
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)
    startRequestId = call_inputs.get("startRequestId", None)

    # sampler = PLMSSampler(model)
    sampler = DDIMSampler(model)

    opt = {
        "n_iter": 1,
        "C": 4,
        "H": 512,
        "W": 512,
        "f": 8,
        "ddim_steps": 50,
        "ddim_eta": 0.0,
        "scale": 7.5,
        "n_samples": 1,
        "skip_save": False,
        "seed": 1,
    }

    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

    prompt = model_inputs.get("prompt", None)
    batch_size = 1
    data = batch_size * [prompt]
    start_code = None

    seed_everything(opt["seed"])

    # precision_scope = autocast if opt.precision == "autocast" else nullcontext
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt["n_iter"], desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt["scale"] != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt["C"], opt["H"] // opt["f"], opt["W"] // opt["f"]]
                        samples_ddim, _ = sampler.sample(
                            S=opt["ddim_steps"],
                            conditioning=c,
                            batch_size=opt["n_samples"],
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt["scale"],
                            unconditional_conditioning=uc,
                            eta=opt["ddim_eta"],
                            x_T=start_code,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_samples_ddim = (
                            x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        )

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(
                            x_checked_image
                        ).permute(0, 3, 1, 2)

                        if not opt["skip_save"]:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                # img.save(
                                #     os.path.join("samples", f"{base_count:05}.png")
                                # )
                                buffered = BytesIO()
                                img.save(buffered, format="JPEG")
                                image_base64 = base64.b64encode(
                                    buffered.getvalue()
                                ).decode("utf-8")

                                # base_count += 1

                toc = time.time()
    return {"image_base64": image_base64}
