FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY python_envs/py3.12_agent/requirements.txt /app/
COPY code/Agent_API_endpoint/app.py /app/

RUN apt update
RUN apt install python3-full -y
RUN apt install python3-venv -y
RUN python3 -m venv venv

RUN venv/bin/pip install --no-cache-dir -r requirements.txt

EXPOSE $AGENT_PORT

CMD venv/bin/uvicorn app:app --host 0.0.0.0 --port $AGENT_PORT