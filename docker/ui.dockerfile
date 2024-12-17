FROM python:3.12-slim

WORKDIR /app

COPY python_envs/py3.12_ui/requirements.txt /app/
COPY code/UI/app.py /app/

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE $UI_PORT

CMD ["streamlit", "run", "app.py", "--server.port", "$LLM_UI_PORT"]