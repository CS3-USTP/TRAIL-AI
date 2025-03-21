FROM python:3.12-bookworm

WORKDIR /app

COPY requirements.txt .

RUN python3.12 -m venv venv && venv/bin/pip install -r requirements.txt

COPY . .

RUN venv/bin/python setup.py

EXPOSE 8000

CMD ["venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]