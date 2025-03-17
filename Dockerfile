FROM python:3.12-slim

WORKDIR /app/src

COPY requirements.txt /app/

RUN pip install --default-timeout=100 --no-cache-dir -r /app/requirements.txt

COPY ./src .

COPY ./test /app/

EXPOSE 8000

CMD ["python", "main.py"]
