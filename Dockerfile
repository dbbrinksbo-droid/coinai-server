FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Flask skal lytte på Railway's port
ENV PORT=8080

CMD ["python3", "server_v2.py"]

