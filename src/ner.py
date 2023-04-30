"""
Functions for named entity recognition
"""
from collections import defaultdict, Counter
from tqdm import tqdm
from pandas import DataFrame
from pathlib import Path
from src.normalize import normalize_pipeline
import json
import numpy as np
from src.data import SetEncoder
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
from dataclasses import dataclass, asdict

@dataclass
class SearchResults:
    noun_score_threshold:float
    noun_query_ratio_threshold:float
    noun_query_count_threshold:float
    evidence_count_per_noun_threshold:float
    coverage:float
    n_shortlist:int
    avg_shortlist:float

    to_dict = asdict


def noun_cond(token) -> str:
    """
    Conditions for entity extraction

    Returns (bool)
    """
    if token.is_stop or token.is_space:
        return False

    if token.pos_ in ["NOUN", "PROPN"]:
        return True

    return False


def train_noun_relations(
    dataset_df:DataFrame,
    nlp,
    save_path:Path=None
) -> defaultdict:
    """
    For a given dataset (as DataFrame), discover all the nouns within
    then train a co-occurrence relation between the claim and evidence pair.
    """
    print("Train noun relations")

    # Cumulator
    nouns_rel = defaultdict(Counter)

    for i, row in tqdm(
        iterable= dataset_df
            .reset_index()
            .iterrows(),
        desc="rows",
        total=dataset_df.shape[0]
    ):
        # Get text and apply pipeline
        claim_text = row["claim_text"]
        evidence_text = row["evidence_text"]

        claim_text = normalize_pipeline(claim_text)
        evidence_text = normalize_pipeline(evidence_text)

        claim_doc = nlp(claim_text)
        evidence_doc = nlp(evidence_text)

        claim_nouns = {token.lemma_ for token in claim_doc if noun_cond(token)}
        evidence_nouns = {token.lemma_ for token in evidence_doc if noun_cond(token)}
        paired_nouns = claim_nouns.union(evidence_nouns)

        # todo count related nouns, only include cross referenced relations
        # ? include co-related nouns within the same passage?
        """
        Count related nouns through cross relation
        """
        cross_relations = [
            (claim_nouns, evidence_nouns),
            (evidence_nouns, claim_nouns)
        ]

        for this_nouns, other_nouns in cross_relations:
            for this_noun in this_nouns:
                for other_noun in other_nouns:
                    if this_noun == other_noun:
                        # Do not count self relation
                        continue
                    nouns_rel[this_noun][other_noun] += 1

            continue

        continue

    # Save the data
    if save_path is not None:
        with open(save_path, mode="w") as f:
            json.dump(obj=nouns_rel, fp=f, cls=SetEncoder)

    return


def get_evidence_by_noun(
    evidence_path:Path,
    nlp,
    save_path:Path=None,
    threshold:int=1000
) -> defaultdict:
    """
    For a given evidence dataset, search all the nouns and create a mapping
    of evidences in which the noun occurred.
    """
    print("For each noun, map an evidence-id")

    with open(evidence_path, mode="r") as f:
        evidences = json.load(fp=f)
        print(f"Loaded evidences n={len(evidences)}")

    # Cumulator
    noun_evidences = defaultdict(set)

    for evidence, evidence_text in tqdm(
        iterable=evidences.items(),
        desc="evidence",
        total=len(evidences)
    ):
        # Get text and apply pipeline
        evidence_text = normalize_pipeline(evidence_text)
        evidence_doc = nlp(evidence_text)
        evidence_nouns = {token.lemma_ for token in evidence_doc if noun_cond(token)}

        # Cumulate the data
        for noun in evidence_nouns:
            noun_evidences[noun].add(evidence)

        continue

    # Save the data
    if save_path is not None:
        with open(save_path, mode="w") as f:
            json.dump(obj=noun_evidences, fp=f, cls=SetEncoder)

    return


def view_claim_noun_phrases(
    dataset_df:DataFrame,
    nlp
) -> defaultdict:
    """
    View the nouns extracted by the pipeline.
    """
    print("View nouns in pipeline")

    for _, row in tqdm(
        iterable= dataset_df
            .reset_index()
            .iterrows(),
        desc="rows",
        total=dataset_df.shape[0],
        leave=False
    ):
        # Get text and apply pipeline
        claim_text = row["claim_text"]
        evidence_text = row["evidence_text"]

        claim_text = normalize_pipeline(claim_text)
        evidence_text = normalize_pipeline(evidence_text)

        claim_doc = nlp(claim_text)
        evidence_doc = nlp(evidence_text)

        claim_nouns = {token.lemma_ for token in claim_doc if noun_cond(token)}
        evidence_nouns = {token.lemma_ for token in evidence_doc if noun_cond(token)}

        print(claim_text)
        print(claim_nouns)
        print(evidence_text)
        print(evidence_nouns)
        print("\n")
        continue

    return

def retrieve_claim_evidence_by_noun(
    claim_path:Path,
    noun_evidences_path:Path,
    noun_rel_path:Path,
    nlp,
    noun_score_threshold:float = 0.01,
    noun_query_ratio_threshold:float = 0.12,
    noun_query_count_threshold:int = 3,
    evidence_count_per_noun_threshold:int = 2000,
    save_path:Path=None,
    verbose=True
) -> defaultdict:
    """
    Retrieves evidence passages for a given claim by matching nouns
    within the claim along with related nouns discovered during training.
    """

    with open(claim_path, mode="r") as f:
        claims = json.load(fp=f)    # Dict[str, Dict[str, List[str]]]

    with open(noun_evidences_path, mode="r") as f:
        noun_evidences = json.load(fp=f)    # Dict[str, List[str]]

    with open(noun_rel_path, mode="r") as f:
        noun_rel = json.load(fp=f)  # Dict[str, Dict[str, int]]

    if verbose:
        print("Retrieve evidence by noun occurrence and relation")
        print(f"Loaded claims n={len(claims)}")
        print(f"Loaded noun evidences n={len(noun_evidences)}")
        print(f"Loaded noun relations n={len(noun_rel)}")

    # Maximum relation count for all nouns
    max_all_rel = max([max(rel.values()) for rel in noun_rel.values()])

    # Cumulator
    claim_evidences = defaultdict(set) # Dict[str, set]

    for claim, claim_data in tqdm(
        iterable=claims.items(),
        desc="claims",
        total=len(claims),
        disable=not verbose
    ):
        # Get text and apply pipeline
        claim_text = claim_data["claim_text"]
        claim_text = normalize_pipeline(claim_text)
        claim_doc = nlp(claim_text)
        claim_nouns = {token.lemma_ for token in claim_doc if noun_cond(token)}

        # Make a copy of the noun_evidences for lookup
        evidences_lookup = noun_evidences.copy()

        # For each noun in claim_nouns,
        #   look up noun relations to get related_nouns.
        for noun in claim_nouns:

            # Add evidences for the noun to the cumulator
            # Delete the noun from the lookup for this claim
            if noun in evidences_lookup.keys():
                evidences = evidences_lookup.get(noun)
                if len(evidences) < evidence_count_per_noun_threshold:
                    claim_evidences[claim].update(set(evidences))
                    del evidences_lookup[noun]

            # Get related nouns, firstly check if we have a relation
            if not noun in noun_rel.keys():
                continue

            # Get related nouns and check if it meets the threshold
            related_nouns = noun_rel.get(noun)
            max_rel = max(related_nouns.values())
            noun_score = max_rel / max_all_rel
            if noun_score < noun_score_threshold:
                continue

            # For each related_noun
            for related_noun, n_rel in related_nouns.items():
                # Check if it has evidences
                if not related_noun in evidences_lookup.keys():
                    continue

                # If it meets the ratio threshold
                noun_query_ratio = n_rel / max_rel
                if noun_query_ratio < noun_query_ratio_threshold:
                    continue

                # If it meets the count threshold
                if n_rel < noun_query_count_threshold:
                    continue

                # Retrieve evidences for the related_noun
                # If the noun has too many evidences attached (eg, it is not
                # specific enough), ignore it
                evidences = evidences_lookup.get(related_noun, [])
                if len(evidences) < evidence_count_per_noun_threshold:
                    # Use and delete the related_noun from the lookup
                    claim_evidences[claim].update(set(evidences))
                    del evidences_lookup[related_noun]

        continue

    if verbose:
        print("Creating shortlist of evidences")
    shortlisted_evidences = set()
    for evidences in claim_evidences.values():
        shortlisted_evidences.update(evidences)

    if save_path is not None:
        if verbose:
            print(f"Saving to {save_path}")
        with open(save_path, mode="w") as f:
            json.dump(obj=claim_evidences, fp=f, cls=SetEncoder)
        with open(save_path.with_name(f"shortlist_{save_path.name}"), mode="w") as f:
            json.dump(obj=shortlisted_evidences, fp=f, cls=SetEncoder)

    # Statistics
    n_average_per_claim = np.mean([
        len(evidences) for evidences in claim_evidences.values()
    ])
    n_shortlisted = len(shortlisted_evidences)

    if verbose:
        print(f"Average number of evidences per claim: {n_average_per_claim:1f}")
        print(f"Number of shortlisted evidences: {n_shortlisted}")
    return shortlisted_evidences, n_average_per_claim, n_shortlisted


def search_eval(job):
    params, nlp, coverage_positives, retrieve_claim_evidence_by_noun = job

    #! Modify also grid_search_thresholds::validation_claims
    shortlist, avg_shortlist, n_shortlist = retrieve_claim_evidence_by_noun(
        claim_path=Path("./data/dev-claims.json"),
        noun_evidences_path=Path("./result/ner/evidence_by_noun.json"),
        noun_rel_path=Path("./result/ner/train_noun_relations.json"),
        nlp=nlp,
        save_path=None,
        verbose=False,
        **params
    )

    # Calculate coverage
    retrieved_positives = coverage_positives.intersection(shortlist)
    coverage = round(len(retrieved_positives) / len(coverage_positives), 2)

    results = SearchResults(
        coverage=coverage,
        n_shortlist=n_shortlist,
        avg_shortlist=avg_shortlist,
        **params
    )
    return results

def grid_search_thresholds():

    import spacy
    nlp = spacy.load("en_core_web_sm")

    SEARCH_SAVE_PATH = Path("./result/ner/retrieval_search_results.json")

    #! Modify also search_eval::claim_path
    with open("./data/dev-claims.json", mode="r") as f:
        validation_claims = json.load(f)

    coverage_positives = {
        evidence
        for claim_dict in validation_claims.values()
        for evidence in claim_dict["evidences"]
    }
    n_coverage_positives = len(coverage_positives)
    print("coverage_positives: ", n_coverage_positives)

    threshold_grid = ParameterGrid(param_grid={
        "noun_score_threshold": np.linspace(start=0.001, stop=0.99, num=20),
        "noun_query_ratio_threshold": np.linspace(start=0.1, stop=0.9, num=10),
        "noun_query_count_threshold": np.linspace(start=1, stop=12, num=6),
        "evidence_count_per_noun_threshold": np.linspace(start=100, stop=10000, num=10),
    })

    jobs = [
        (params, nlp, coverage_positives, retrieve_claim_evidence_by_noun)
        for params in threshold_grid
    ]

    with Pool(processes=8) as pool:
        with tqdm(total=len(jobs)) as pbar:
            search_results = []
            for result in pool.imap(func=search_eval, iterable=jobs):
                search_results.append(result)
                pbar.update()

    with open(SEARCH_SAVE_PATH, mode="w") as f:
        json.dump([x.to_dict() for x in search_results], f)

    return

if __name__ == "__main__":
    grid_search_thresholds()
    pass