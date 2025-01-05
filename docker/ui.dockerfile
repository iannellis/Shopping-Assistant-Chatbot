FROM python:3.12-slim

WORKDIR /app

COPY python_envs/py3.12_ui/requirements.txt /app/

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY code/UI/ui2.py /app/

EXPOSE $UI_PORT

CMD streamlit run ui2.py