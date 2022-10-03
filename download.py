import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

CONFIG_FILE = "configs/stable-diffusion/v1-inference.yaml"


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # do this elsewhere
    # model.cuda()
    # model.eval()
    return model


def main():
    config = OmegaConf.load(CONFIG_FILE)
    model = load_model_from_config(config, "/models/model-epoch07-float16.ckpt")


if __name__ == "__main__":
    main()
