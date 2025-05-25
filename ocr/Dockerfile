# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the app files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the API
EXPOSE 5003

# Run the FastAPI app
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]
