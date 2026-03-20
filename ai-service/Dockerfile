FROM python:3.11-slim

WORKDIR /app

# System deps for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MobileNetV2 weights at build time so the container
# starts instantly (no internet required at runtime)
RUN python -c "\
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2; \
MobileNetV2(weights='imagenet'); \
print('Weights cached.')"

COPY . .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "app:app"]
