FROM continuumio/miniconda3:4.12.0

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive

# RUN conda install python=3.8.5 && conda clean -a -y
# RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch && conda clean -a -y

RUN git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion && cd repositories/stable-diffusion && git reset --hard 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc
WORKDIR /repositories/stable-diffusion
RUN conda update -n base -c defaults conda
RUN conda env create -f environment.yaml

# This will download missing models
RUN conda run --no-capture-output -n ldm \
  python scripts/txt2img.py || true

#RUN mkdir -p models/ldm/stable-diffusion-v1
RUN mkdir -p /models
RUN wget --show-progress --progress=bar:force https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/model-epoch07-full.ckpt -P /models

ADD txt2img.py .
ADD download.py .
RUN conda run --no-capture-output -n ldm \
  python download.py

##==22.6.2
RUN conda install -n ldm -c conda-forge sanic==22.6.2


ADD server.py .
ADD app.py .

EXPOSE 8000

# Runtime vars (for init and inference); fork / downstream specific.
# ADD APP_VARS.py .
#ENV SEND_URL=""
#ENV SIGN_KEY=""
CMD conda run --no-capture-output -n ldm python -u server.py
