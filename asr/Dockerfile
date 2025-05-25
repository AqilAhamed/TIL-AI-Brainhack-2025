# Use an official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean

# Copy your requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your ASR code into the image
COPY . .
COPY src/asr_server.py .
COPY src/asr_manager.py .

COPY finetuned-wav2vec2/ /app/finetuned-wav2vec2/
ENV MODEL_DIR=/app/finetuned-wav2vec2

#COPY model.onnx /app/model.onnx
#COPY processor .

# Expose port 5001 to the host
EXPOSE 5001

# Start the ASR FastAPI server on port 5001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]


