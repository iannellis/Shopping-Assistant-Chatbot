FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY python_envs/py3.12_llm_db/requirements.txt /app/
COPY code/LLM_DB_API_endpoint/app.py /app/

RUN apt update
RUN apt install python3-full -y
RUN apt install python3-venv -y
RUN python3 -m venv venv

RUN venv/bin/pip install --no-cache-dir -r requirements.txt

EXPOSE $LLM_DB_PORT

CMD venv/bin/uvicorn app:app --host 0.0.0.0 --port $LLM_DB_PORT