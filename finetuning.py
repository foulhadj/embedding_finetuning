import os
import json
import warnings
from typing import List, Dict

import nest_asyncio
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# Ignorer les avertissements spécifiques
warnings.filterwarnings("ignore", message="Using the `WANDB_DISABLED` environment variable is deprecated")

# Configuration globale
BATCH_SIZE = 20
EPOCHS = 5
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
REPO_NAME = "finetuned_arctic_aichor_4"


def setup_environment():
    """Configure l'environnement pour l'exécution."""
    nest_asyncio.apply()
    login(token=os.getenv('HUGGING_TOKEN'), add_to_git_credential=False)
    os.environ["WANDB_DISABLED"] = "true"


def load_dataset(file_path: str) -> Dict:
    """Charge un dataset à partir d'un fichier JSON."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_input_examples(queries: Dict, corpus: Dict, relevant_docs: Dict) -> List[InputExample]:
    """Crée une liste d'InputExample à partir des données du dataset."""
    return [
        InputExample(texts=[query, corpus[relevant_docs[query_id][0]]])
        for query_id, query in queries.items()
    ]


def collate_fn(batch):
    """Transforme une liste d'InputExample en un format utilisable."""
    texts = [example.texts for example in batch]
    return {"texts": [t.to(device) if isinstance(t, torch.Tensor) else t for t in texts]}


def train_model(model, train_loader, val_dataset, device):
    """Entraîne le modèle avec les données fournies."""
    # Définir la fonction de perte
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    inner_train_loss = MultipleNegativesRankingLoss(model).to(device)
    train_loss = MatryoshkaLoss(
        model,
        inner_train_loss,
        matryoshka_dims=matryoshka_dimensions,
    ).to(device)

    # Définir l'évaluateur
    evaluator = InformationRetrievalEvaluator(
        val_dataset['questions'],
        val_dataset['corpus'],
        val_dataset['relevant_contexts']
    )

    # Configuration de l'entraînement
    warmup_steps = int(len(train_loader) * EPOCHS * 0.1)

    # Entraînement
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=None,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=50,
    )


def save_model(model):
    """Sauvegarde le modèle sur Hugging Face Hub."""
    model.save_to_hub(
        REPO_NAME,
        organization=None,
        private=True,
        commit_message="Modèle fine-tuné sur aichor",
    )


def main():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    print(f"Using device: {device}")

    setup_environment()

    # Charger le modèle
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True).to(device)

    # Charger et préparer les données d'entraînement
    train_dataset = load_dataset("input_data/training_dataset.jsonl")
    train_examples = create_input_examples(
        train_dataset['questions'],
        train_dataset['corpus'],
        train_dataset['relevant_contexts']
    )
    train_loader = DataLoader(train_examples, batch_size=BATCH_SIZE, drop_last=True, collate_fn=collate_fn)

    # Charger les données de validation
    val_dataset = load_dataset("input_data/val_dataset.jsonl")

    # Entraîner le modèle
    train_model(model, train_loader, val_dataset, device)

    # Sauvegarder le modèle
    save_model(model)


if __name__ == "__main__":
    main()
