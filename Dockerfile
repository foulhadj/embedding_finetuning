FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN mkdir -p /opt/app/src
WORKDIR /opt/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers==2.2.2 huggingface_hub==0.13.4


# Copy the application code into the container
COPY finetuning.py /opt/app/src/
COPY manifest /opt/app/src/
COPY input_data /opt/app/src/input_data

# Set environment variables
ENV WANDB_DISABLED=true
#ENV DNS_SERVER 8.8.8.8 8.8.4.4

CMD ["python", "finetuning.py", "--report_to", "none"]

# # Use a Python base image
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# # Set the working directory inside the container
# RUN mkdir -p /opt/app/src
# WORKDIR /opt/app

# # Copy the requirements file into the container
# COPY requirements.txt .

# # Update pip and install the dependencies
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy the application code into the container
# COPY finetuning.py .
# COPY input_data ./input_data

# # Set environment variables
# ENV WANDB_DISABLED=true

# # Définir la commande par défaut pour exécuter l'application
# CMD ["python", "finetuning.py", "--report_to", "none"]


# # Use a Python base image
# FROM python:3.11-slim-buster

# WORKDIR /opt/app

# # Copy the requirements file into the container
# COPY requirements.txt .

# # Install the dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code into the container
# COPY finetuning.py .
# COPY input_data ./input_data
# # Set environment variables
# ENV WANDB_DISABLED=true

# CMD ["python", "finetuning.py", "--report_to", "none"]