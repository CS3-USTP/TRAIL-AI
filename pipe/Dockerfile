FROM python:3.12-bookworm

WORKDIR /app

COPY requirements-cpu.txt .

RUN python3.12 -m venv venv 

RUN venv/bin/pip install -r requirements-cpu.txt

COPY . .

RUN venv/bin/python setup.py

RUN venv/bin/python scripts/download.py

EXPOSE 8000

CMD ["venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]