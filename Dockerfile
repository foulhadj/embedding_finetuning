# Use a Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /opt/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY finetuning.py .
COPY input_data ./input_data

# Set environment variables
ENV WANDB_DISABLED=true

# Command to run the application
CMD ["python", "finetuning.py"]