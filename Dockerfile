# Utiliser l'image de base PyTorch avec CUDA 11.6
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Définir le répertoire de travail
WORKDIR /opt/app

# Créer un répertoire pour le code source et y copier les fichiers
RUN mkdir -p /opt/app/src

# Copier le fichier requirements.txt dans l'image Docker
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'application dans l'image Docker
COPY finetuning.py .
COPY input_data ./input_data

# Définir la variable d'environnement WANDB_DISABLED
ENV WANDB_DISABLED=true

# Définir la commande par défaut pour exécuter l'application
CMD ["python", "finetuning.py", "--report_to", "none"]