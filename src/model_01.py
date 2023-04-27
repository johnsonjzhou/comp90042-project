"""
Evidence retrieval using Siamese BERT classifier.

Ref:
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://huggingface.co/blog/how-to-train-sentence-transformers
- https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/SoftmaxLoss.py
- https://github.com/UKPLab/sentence-transformers/blob/42ab80a122ad521a4f9055f5530a954d29d232ce/sentence_transformers/readers/InputExample.py
"""
import torch
from torch.utils.data import Dataset
from torch import nn
from sentence_transformers import SentenceTransformer, InputExample, \
    LoggingHandler, util
import spacy
from pathlib import Path
from typing import Callable
import pandas as pd
from pandas import DataFrame
import numpy as np
import json
from src.torch_utils import get_torch_device
from src.spacy_utils import process_sentence
from src.data import create_claim_output, load_from_json
from tqdm import tqdm
import random
from dataclasses import dataclass
import os
from pathlib import Path
import logging


class ClaimEvidencePairDataset(Dataset):
    """
    A dataset consisting of claim evidence pairs with labels
    """

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
            self.data = json.load(fp=f)
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> InputExample:
        # Fetch the required data rows
        data = self.data[idx]
        label = data["related"]
        texts = [data["claim_text"], data["evidence_text"]]

        # Run preprocessing if a preprocessing function is provided
        if self.preprocess_func is not None:
            for i, text in enumerate(texts):
                if not text.startswith("[CLS]"):
                    texts[i] = self.preprocess_func(
                        text,
                        nlp=self.nlp,
                        bert_tokens=self.bert_tokens
                    )

        # Convert to InputExample
        return InputExample(texts=texts, label=label)

@dataclass
class ClaimEvidencePair:
    claim_id:str
    evidence_id:str
    label:int = None


class ClaimEvidenceDataset(Dataset):
    """
    A dataset consisting of claim evidence pairs with labels
    """

    NEG_SAMP_STRATEGIES = ["random", "related_random"]

    def __init__(
        self,
        claims_json:Path,
        evidence_json:Path,
        negative_sample_strategy:str="related_random",
        negative_sample_size:int=0,
        random_seed:int=42,
        verbose:bool=True,
        preprocess_func:Callable=None,
        nlp=None,
        bert_tokens:bool=True
    ):
        """
        Args:
            claims_json (Path): Path to the claims json file.
            evidence_json (Path): Path to the evidence json file.
            negative_sample_strategy (str, optional): \
                Either "random" or "related_random". \
                "random" will include random hard negatives from the \
                general evidence corpus. \
                "related_random" will additionally add random hard negatives \
                from positives matched in the claims. \
                Defaults to "related_random".
            negative_sample_size (int, optional): \
                `-1`: include all evidences. \
                `0`: include an equal number of hard negatives as positives. \
                `>0`: include a defined number of hard negatives.\
                Defaults to 0.
            random_seed (int, optional): For random sampling. Defaults to 42.
            verbose (bool, optional): Show a tqdm progress bar. Defaults to True.
            preprocess_func (Callable): Function for preprocessing data. \
                Default to `None`.
            nlp: A spaCy language model.
            bert_tokens: Whether to include BERT tokens
        """
        self.verbose = verbose
        random.seed(a=random_seed)

        # Custom preprocessing methods
        self.preprocess_func = preprocess_func
        self.nlp = nlp
        self.bert_tokens = bert_tokens

        # Load data from json
        with open(claims_json, mode="r") as f:
            self.claims = json.load(fp=f)
        with open(evidence_json, mode="r") as f:
            self.evidence = json.load(fp=f)

        # Negative sampling
        assert negative_sample_strategy in self.NEG_SAMP_STRATEGIES
        self.negative_sample_strategy = negative_sample_strategy
        self.negative_sample_size = negative_sample_size

        # Generate data
        self.related_evidence_ids = self.__extract_related_evidence_ids()
        self.distant_evidence_ids = \
            set(self.evidence.keys()).difference(self.related_evidence_ids)
        self.data = self.__generate_data()
        return

    def __extract_related_evidence_ids(self) -> set:
        related_evidence_ids = set()
        for claim in self.claims.values():
            if "evidences" in claim.keys():
                related_evidence_ids.update(claim["evidences"])
        return related_evidence_ids

    def __generate_data(self):
        print(
            "Generate claim-evidence pair with",
            self.negative_sample_strategy,
            "strategy",
            f"n={self.negative_sample_size}"
        )
        data = []
        for claim_id, claim in tqdm(
            iterable=self.claims.items(),
            desc="claims",
            disable=not self.verbose
        ):
            # Check if we have evidences supplied, this will inform
            # whether this is for training
            is_training = "evidences" in claim.keys()
            n_evidences = 0
            pos_evidence_ids = set()

            # Get positive samples from evidences with label=1
            if is_training:
                pos_evidence_ids.update(claim["evidences"])
                n_evidences = len(pos_evidence_ids)

                for evidence_id in pos_evidence_ids:
                    data.append(ClaimEvidencePair(
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        label=1
                    ))

            # Apply negative sampling rules ----------------------------------

            # Hard negatives from related evidences
            if self.negative_sample_strategy == "related_random":
                if not is_training:
                    raise ValueError(
                        "Cannot do related random with no evidence relations"
                    )

                # Set the number of sample to get, this could be set by
                # the parameter or equal to the number of truth evidences
                sample_size = n_evidences \
                    if self.negative_sample_size == 0 \
                    else self.negative_sample_size

                # Get related negative evidences
                # Exclude those that are positives
                related_neg_evidence_ids = \
                    self.related_evidence_ids.difference(pos_evidence_ids)

                # If sample_size < 0, this means all available
                # Else, randomly sample
                if sample_size > 0:
                    related_neg_evidence_ids = random.sample(
                        population=related_neg_evidence_ids,
                        k=min(sample_size, len(related_neg_evidence_ids))
                    )

                # Generate claim and related negative evidence pairs
                for evidence_id in related_neg_evidence_ids:
                    data.append(ClaimEvidencePair(
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        label=0
                    ))

            # Hard negatives from distant evidences
            if self.negative_sample_strategy in ["related_random", "random"]:
                # Set the number of sample to get, this could be set by
                # the parameter or equal to the number of truth evidences
                sample_size = n_evidences \
                    if self.negative_sample_size == 0 \
                    else self.negative_sample_size

                # Get related negative evidences
                # Exclude those that are positives
                distant_neg_evidence_ids = \
                    self.distant_evidence_ids.difference(pos_evidence_ids)

                # If sample_size < 0, this means all available
                # Else, randomly sample
                if sample_size > 0:
                    distant_neg_evidence_ids = random.sample(
                        population=distant_neg_evidence_ids,
                        k=min(sample_size, len(distant_neg_evidence_ids))
                    )

                # Generate claim and distant negative evidence pairs
                for evidence_id in distant_neg_evidence_ids:
                    data.append(ClaimEvidencePair(
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        label=0
                    ))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> InputExample:
        # Fetch the required data rows
        data = self.data[idx]
        label = data.label
        claim_id = data.claim_id
        evidence_id = data.evidence_id
        texts = [
            self.claims[claim_id]["claim_text"],
            self.evidence[evidence_id]
        ]

        # Run preprocessing if a preprocessing function is provided
        if self.preprocess_func is not None:
            for i, text in enumerate(texts):
                if not text.startswith("[CLS]"):
                    texts[i] = self.preprocess_func(
                        text,
                        nlp=self.nlp,
                        bert_tokens=self.bert_tokens
                    )

        # Convert to InputExample
        return InputExample(texts=texts, label=label)


def run_inference(
    name:str,
    model:SentenceTransformer,
    claims:dict,
    evidence:dict,
    scorer:Callable,
    threshold:float,
    output_path:Path,
    batch_size:int=64,
    device=None,
    verbose:bool=True
):
    # Save paths
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    claim_embedding_path = output_path.joinpath(f"{name}_claim_embeddings.pt")
    evidence_embedding_path = \
        output_path.joinpath(f"evidence_embeddings.pt")
    output_json = output_path.joinpath(f"{name}_evidence_predictions.json")

    # Get lists of texts
    claim_texts = [claim["claim_text"] for claim in claims.values()]
    evidence_texts = list(evidence.values())

    # Create embeddings of claim texts
    print(f"Generate claim embeddings n={len(claim_texts)}")
    if os.path.exists(claim_embedding_path):
        # Load it from file if we have already created it
        with open(claim_embedding_path, mode="rb") as f:
            claim_emb = torch.load(f=f, map_location=device)
            print("Loaded claim embeddings from file")
    else:
        # Create the embeddings and save it to file
        claim_emb = model.encode(
            sentences=claim_texts,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=batch_size,
            device=device
        )
        with open(claim_embedding_path, mode="wb") as f:
            torch.save(obj=claim_emb, f=f)
            print("Saved claim embeddings to file")

    # Create embeddings of evidence texts
    print(f"Generate evidence embeddings n={len(evidence_texts)}")
    if os.path.exists(evidence_embedding_path):
        # Load it from file if we have already created it
        with open(evidence_embedding_path, mode="rb") as f:
            evidence_emb = torch.load(f=f, map_location=device)
            print("Loaded evidence embeddings from file")
    else:
        # Create the embeddings and save it to file
        evidence_emb = model.encode(
            sentences=evidence_texts,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=batch_size,
            device=device
        )
        with open(evidence_embedding_path, mode="wb") as f:
            torch.save(obj=evidence_emb, f=f)
            print("Saved evidence embeddings to file")

    # Calculate the scores
    print("Calculate scores")
    scores = scorer(a=claim_emb, b=evidence_emb)

    # For debug purposes keep track of number of evidences retrieved
    n_evidence_retrieved = list()

    # Holder for the output objects
    output = dict()

    # Go through each claim and retrieve up to 5 top scoring evidences
    print("Retrieve top scoring evidences")
    for claim_id, evidence_scores in tqdm(
        iterable=zip(claims.keys(), scores),
        desc="claims",
        disable=not verbose
    ):
        claim_text = claims[claim_id]["claim_text"]
        df_scores = DataFrame(
            data={
                "id": list(evidence.keys()),
                "score": evidence_scores.cpu().numpy()
            }
        )

        # Get related evidences by applying cutoff from threshold
        related_evidences = (
            df_scores[df_scores["score"] > threshold]
            .sort_values(by="score", ascending=False)
        )

        # Keep track of how many evidences retrieved
        n_evidence_retrieved.append(related_evidences.shape[0])

        # Get a maximum of the top 5 scoring evidences
        related_evidences = related_evidences.iloc[:5]

        # Create the output
        output.update(create_claim_output(
            claim_id=claim_id,
            claim_text=claim_text,
            evidences=related_evidences["id"].to_list()
        ))
        continue

    print(f"Average retrievals = {np.mean(n_evidence_retrieved):2f}")

    # Write output to file
    with open(output_json, mode="w") as f:
        json.dump(obj=output, fp=f, ensure_ascii=True)

    print("Done!")
    return

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

        print(data)
    pass

def test_data_new():
    # Create spacy language model
    nlp = spacy.load("en_core_web_trf")

    dataset = ClaimEvidenceDataset(
        claims_json="./data/train-claims.json",
        evidence_json="./data/evidence.json",
        negative_sample_strategy="related_random",
        negative_sample_size=0,
        # preprocess_func=process_sentence,
        preprocess_func=None,
        nlp=nlp,
        bert_tokens=True
    )

    for i, data in enumerate(dataset):
        if i == 3:
            break

        print(data)

    print(len(dataset))
    pass


def dev_inference():
    """
    This is equivalent to the jupyter notebook
    """
    # Paths
    ROOT_DIR = Path.cwd()
    MODEL_PATH = ROOT_DIR.joinpath("./result/models/*")
    OUTPUT_PATH = ROOT_DIR.joinpath("./result/inference")

    # Names
    model_save_path = MODEL_PATH.with_name(f"model_01_base_e5_equal_neg")
    inference_output_path = OUTPUT_PATH.joinpath(model_save_path.name)

    # Logging
    logging.basicConfig(format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()]
    )

    # Load datasets
    data_names = ["dev-claims", "test-claims-unlabelled", "evidence"]
    dev_claims, test_claims, all_evidence = load_from_json(data_names)

    # Load model from file
    torch_device = get_torch_device()
    model = SentenceTransformer(
        model_name_or_path=model_save_path,
        device=torch_device
    )

    # Run inference
    run_inference(
        name="dev",
        model=model,
        claims=dev_claims,
        evidence=all_evidence,
        scorer=util.cos_sim,
        threshold=0.6251,
        output_path=inference_output_path,
        batch_size=64,
        device=torch_device,
        verbose=True
    )
    return


if __name__ == "__main__":
    # test_data()
    # test_data_new()
    dev_inference()
    pass