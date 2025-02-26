# Use a Python base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory inside the container
RUN mkdir -p /opt/app/src
WORKDIR /opt/app

# Copy the requirements file into the container
COPY requirements.txt .

# Update pip and install the dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY finetuning.py .
COPY input_data ./input_data

# Set environment variables
ENV WANDB_DISABLED=true

# Command to run the application
CMD ["python", "finetuning.py"]