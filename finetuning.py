import os
import json
import nest_asyncio
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# Configuration
nest_asyncio.apply()
login(token=os.getenv('HUGGING_TOKEN'), add_to_git_credential=True)
os.environ["WANDB_DISABLED"] = "true"

BATCH_SIZE = 20
EPOCHS = 5

# Load model from the hub
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

# Load training dataset
with open("input_data/training_dataset.jsonl", "r") as f:
    train_dataset = json.load(f)

corpus = train_dataset['corpus']
queries = train_dataset['questions']
relevant_docs = train_dataset['relevant_contexts']

examples = [
    InputExample(texts=[query, corpus[relevant_docs[query_id][0]]])
    for query_id, query in queries.items()
]

loader = DataLoader(examples, batch_size=BATCH_SIZE)

# Load validation dataset
with open("input_data/val_dataset.jsonl", "r") as f:
    val_dataset = json.load(f)

val_corpus = val_dataset['corpus']
val_queries = val_dataset['questions']
val_relevant_docs = val_dataset['relevant_contexts']

# Define loss function
matryoshka_dimensions = [768, 512, 256, 128, 64]
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model,
    inner_train_loss,
    matryoshka_dims=matryoshka_dimensions,
)

# Define evaluator
evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant_docs)

# Training configuration
warmup_steps = int(len(loader) * EPOCHS * 0.1)
model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=None,
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50,
)

# Save the model to Hugging Face Hub
REPO_NAME = "finetuned_arctic_aichor"
model.save_to_hub(
    REPO_NAME,
    organization=None,
    private=True,
    commit_message="Modèle fine-tuné sur aichor",
)
