# Utiliser l'image de base compatible CUDA (NVIDIA A100 et autres GPU)
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu20.04

# Installer les dépendances de base (Python, pip, etc.)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les bibliothèques nécessaires via pip
RUN pip3 install sentence_transformers huggingface_hub torch torchvision torchaudio nest_asyncio xformers datasets

# Copier ton code dans l'image Docker
COPY . /app

# Définir le répertoire de travail
WORKDIR /app

# Commande par défaut pour démarrer l'application
CMD ["python3", "finetuning.py"]