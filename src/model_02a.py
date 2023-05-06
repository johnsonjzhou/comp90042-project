"""
Support for model_02a fast shortlisting
"""
from multiprocessing.pool import Pool
from tqdm import tqdm
import json
from pathlib import Path

import spacy
nlp = spacy.load("en_core_web_sm")

# def info_tag_pipeline(
#     text:str,
#     go_nouns:list = [],
#     return_doc:bool=False
# ) -> List[InfoTag]:
#     text = normalize_pipeline(text)
#     doc = nlp(text)
#     tags = get_info_tags(doc, go_nouns=go_nouns)

#     if return_doc:
#         return tags, doc
#     else:
#         return tags

def get_bidirectional_n_grams(doc, n:int=4):
    fwd_ngrams = [token.lemma_[:n] for token in doc if len(token.lemma_) >= n]
    rev_ngrams = [token.lemma_[-n:] for token in doc if len(token.lemma_) >= n]
    return fwd_ngrams, rev_ngrams

def get_multiprocess_nlp(foo, nlp=nlp):
    text_id, text = foo
    return text_id, nlp(text)

def test_multiprocessing(evidence_path:Path):
    # Load the evidence file
    with open(evidence_path, mode="r") as f:
        evidences = json.load(f)

    evidences_iter = list(evidences.items())

    with Pool(10) as pool:
        doc_results = pool.imap(get_multiprocess_nlp, evidences_iter)

        for result in tqdm(doc_results, desc="evidences", total=len(evidences_iter)):
            continue


if __name__ == "__main__":
    test_multiprocessing(
        evidence_path=Path("./data/evidence.json"),
    )
    pass
