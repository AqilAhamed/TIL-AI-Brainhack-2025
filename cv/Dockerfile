FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system deps for OpenCV & building pycocotools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code & artifacts
COPY . .
COPY src/cv_server.py .
COPY src/cv_manager.py .

COPY weights/fasterrcnn_novice.pth .
ENV WEIGHTS_PATH=/app/weights/fasterrcnn_novice.pth

COPY annotations.json .
ENV ANNOTATIONS_PATH=/app/annotations.json

COPY resnet50-0676ba61.pth .
ENV RESNET_WEIGHTS_PATH=/app/resnet50-0676ba61.pth


# Vertex AI expects you to bind on $PORT (default 8080)
#ARG PORT=5002
#ENV PORT=${PORT}
#EXPOSE ${PORT}
EXPOSE 5002

# Launch Uvicorn on $PORT
#CMD ["sh", "-c", "uvicorn cv_server:app --host 0.0.0.0 --port ${PORT}"]
CMD ["uvicorn", "cv_server:app", "--host", "0.0.0.0", "--port", "5002"]