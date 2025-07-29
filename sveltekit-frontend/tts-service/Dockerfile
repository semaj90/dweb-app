FROM python:3.11-slim
WORKDIR /app
COPY server.py .
RUN pip install --no-cache-dir flask flask-cors requests
ENV PYTHONUNBUFFERED=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
CMD ["python", "server.py"]
