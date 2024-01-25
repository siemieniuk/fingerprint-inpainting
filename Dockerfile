FROM python:3.10-slim

WORKDIR /backend

COPY requirements_docker.txt ./requirements.txt

RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

COPY *.py .
COPY params/ params/
COPY weights/ weights/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]