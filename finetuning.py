import os
import json
import nest_asyncio
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

# Configuration
nest_asyncio.apply()
login(token=os.getenv('HUGGING_TOKEN'), add_to_git_credential=False)
os.environ["WANDB_DISABLED"] = "true"

BATCH_SIZE = 20
EPOCHS = 5

def main():
    # Initialize process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # TensorBoard writer
    writer = SummaryWriter(os.environ.get("AICHOR_TENSORBOARD_PATH", "./runs"))

    # Load model from the hub
    MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

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

    # Distributed sampler
    train_sampler = DistributedSampler(examples)
    loader = DataLoader(examples, batch_size=BATCH_SIZE, sampler=train_sampler)

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

    # Custom training loop
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(loader):
            loss = train_loss(batch)
            loss.backward()
            # Add logging
            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), epoch * len(loader) + batch_idx)
        
        # Evaluation
        if local_rank == 0:
            scores = evaluator(model)
            for metric, value in scores.items():
                writer.add_scalar(f'eval/{metric}', value, epoch)

    writer.close()

    # Save the model to Hugging Face Hub (only on main process)
    if local_rank == 0:
        REPO_NAME = "finetuned_arctic_aichor_3"
        model.module.save_to_hub(
            REPO_NAME,
            organization=None,
            private=True,
            commit_message="Modèle fine-tuné sur aichor",
        )

if __name__ == "__main__":
    main()
