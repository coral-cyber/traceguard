FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output dirs
RUN mkdir -p training/models training/data

# Expose Gradio port
EXPOSE 7860

# Run training then launch app
CMD ["sh", "-c", "python -m training.generate_data && python -m training.train_overseer && python app.py"]
