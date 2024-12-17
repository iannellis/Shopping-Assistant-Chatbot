FROM nvidia/cuda:12.6.3-base-ubuntu24.04

WORKDIR /app

COPY python_envs/py3.12_llm_db/requirements.txt /app/
COPY code/LLM_DB_API_endpoint/app.py /app/

RUN apt update
RUN apt install python3-pip -y

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE $LLM_DB_PORT

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$LLM_DB_PORT"]