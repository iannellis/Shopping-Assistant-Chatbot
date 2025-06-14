FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.11 -y
RUN apt install python3-pip -y

COPY python_envs/py3.11_blip-2/requirements.txt /app/
COPY assets/*.whl /app/

RUN python3.11 -m pip install *.whl 
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY assets/blip-2-processors-pretrain.pkl /app/
COPY assets/blip-2-processors-coco.pkl /app/
COPY code/Blip-2_API_endpoint/* /app/

EXPOSE $BLIP_2_PORT

CMD uvicorn app:app --host 0.0.0.0 --port $BLIP_2_PORT