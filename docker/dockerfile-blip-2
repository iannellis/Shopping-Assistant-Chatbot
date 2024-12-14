FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

WORKDIR /app

COPY python_envs/py3.11_deploy/requirements.txt /app/
COPY assets/*.whl /app/
COPY assets/blip-2-processors.pkl /app/
COPY code/Blip-2_API_endpoint/app.py /app/

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.11 -y
RUN apt install python3-pip -y

RUN python3.11 -m pip install *.whl 
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]