# Use a Python base image
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY finetuning.py .
COPY input_data ./input_data

# Set environment variables
ENV WANDB_DISABLED=true

# Configure Hugging Face login (Optional - consider security implications)
# ARG HUGGING_FACE_API_TOKEN
# ENV HUGGING_TOKEN=${HUGGING_FACE_API_TOKEN}

# Command to run the application
CMD ["python", "finetuning.py"]