import torch

# Afficher la version de torch et la disponibilité de CUDA
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())