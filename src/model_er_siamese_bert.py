"""
Evidence retrieval using Siamese BERT classifier.

Ref:
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import torch
import spacy
from pathlib import Path
from typing import Callable, Dict
import pandas as pd
from src.spacy_utils import process_sentence

class ClaimEvidencePairDataset(torch.utils.data.Dataset):
    """
    A dataset consisting of claim evidence pairs with labels
    """

    # Columns to which preprocessing is applied
    PP_COLS = ["claim_text", "evidence_text"]

    def __init__(
        self, json_file:Path, preprocess_func:Callable=None, nlp=None,
        bert_tokens:bool=True
    ):
        """
        JSON data file should have the following keys:
            - `claim`: claim id
            - `claim_text`: claim text string
            - `evidence`: evidence id
            - `evidence_text`: evidence text string
            - `related`: relation labels as `1/0`

        Args:
            json_file (Path): `Path` to the json data file.
            preprocess_func (Callable): Function for preprocessing data. \
                Default to `None`.
            nlp: A spaCy language model.
            bert_tokens: Whether to include BERT tokens
        """
        self.preprocess_func = preprocess_func
        self.nlp = nlp
        self.bert_tokens = bert_tokens

        # Load data from json, should have shape (n, 5)
        with open(json_file, mode="r") as f:
            self.data = pd.read_json(f, orient="records")
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> Dict[str, str]:
        # Fetch the required data rows
        data = self.data.iloc[idx].to_frame().T

        # Run preprocessing if a preprocessing function is provided
        if self.preprocess_func is not None:
            data[self.PP_COLS] = (
                data[self.PP_COLS]
                .applymap(
                    func=self.preprocess_func,
                    nlp=self.nlp,
                    bert_tokens=True
                )
            )

        # Return data as a dictionary
        return data.to_dict(orient="records")[0]

def test_data():
    # Create spacy language model
    nlp = spacy.load("en_core_web_trf")

    # Path to the json data file
    json_file = \
        Path("./result/train_data/train_claim_evidence_pair_rns.json")

    dataset = ClaimEvidencePairDataset(
        json_file=json_file,
        preprocess_func=process_sentence,
        nlp=nlp,
        bert_tokens=True
    )

    for i, data in enumerate(dataset):
        if i == 3:
            break

        print(f"Pair {i}\n")
        print(f"Claim:\n{data['claim_text']}\n")
        print(f"Evidence:\n{data['evidence_text']}\n")
    pass


if __name__ == "__main__":
    test_data()
    pass