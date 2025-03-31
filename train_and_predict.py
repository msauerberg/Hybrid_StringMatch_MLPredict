import pandas as pd
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    util,
    models,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset


def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return df


def prepare_train_examples(df, add_negative_samples=False):
    train_examples = []
    for _, row in df.iterrows():
        train_examples.append(
            InputExample(texts=[row["input_text"], row["label"]], label=1.0)
        )

    if add_negative_samples:
        unique_labels = df["label"].unique().tolist()
        for _, row in df.iterrows():
            negative_label = random.choice(
                [lbl for lbl in unique_labels if lbl != row["label"]]
            )
            train_examples.append(
                InputExample(texts=[row["input_text"], negative_label], label=0.0)
            )

    return train_examples


def train_model(
    train_examples, val_examples, model_name="emilyalsentzer/Bio_ClinicalBERT", epochs=3
):
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_loss = losses.MultipleNegativesRankingLoss(model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="val-eval"
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=100,
        output_path="substance_extractor_model",
    )
    return model


def predict_substance(input_text, model, reference_list):
    corpus_embeddings = model.encode(
        reference_list, show_progress_bar=True, convert_to_tensor=True
    )
    query_embedding = model.encode(input_text, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
    best_match = reference_list[results[0][0]["corpus_id"]]
    similarity_score = round(results[0][0]["score"], 2)

    return best_match, similarity_score


def encode_reference_list(model, reference_list):
    return model.encode(reference_list, convert_to_tensor=True, show_progress_bar=True)


def predict_substances_batch(df, model, reference_list, output_csv="predictions.csv"):
    """Predict substances for all input_text values in the dataset efficiently."""

    # Precompute reference embeddings once
    reference_embeddings = encode_reference_list(model, reference_list)

    # Encode all input_text in batch
    input_texts = df["input_text"].tolist()
    ID_text = df["ID"].tolist()
    input_embeddings = model.encode(
        input_texts, convert_to_tensor=True, show_progress_bar=True
    )

    predictions = []

    # Perform semantic search in batch
    results = util.semantic_search(input_embeddings, reference_embeddings, top_k=1)

    for i, result in enumerate(results):
        best_match = reference_list[result[0]["corpus_id"]]
        similarity_score = round(result[0]["score"], 2)
        predictions.append(
            {
                "ID": ID_text[i],
                "Original": input_texts[i],
                "Predicted": best_match,
                "Similarity": similarity_score,
            }
        )

    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False, sep=";")
    print(f"Predictions saved to {output_csv}")
