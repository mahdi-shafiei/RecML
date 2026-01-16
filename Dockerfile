# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# This tells Python to look in /app for the 'recml' package
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install system tools if needed (e.g., git)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Force install the specific protobuf version
RUN pip install "protobuf>=6.31.1" --no-deps

# Default command to run the training script
CMD ["python", "recml/examples/dlrm_experiment_test.py"]
