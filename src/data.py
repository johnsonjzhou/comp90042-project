"""
Data I/O functions
"""
import pathlib
import pandas as pd
from pandas import DataFrame
import re
import json
from typing import List
from collections import defaultdict

# Colors for printing to the terminal
# Ref: https://stackoverflow.com/a/287944
class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_from_json(names:list) -> List[dict]:
    DATA_PATH = pathlib.Path("data", "*")
    datasets = list()
    for name in names:
        with open(DATA_PATH.with_name(name).with_suffix(".json")) as json_file:
            dataset = json.load(json_file)
            datasets.append(dataset)
            print(f"Loaded {name}")
    return datasets

def load_as_dataframe(names:list) -> List[DataFrame]:
    """
    Gets the json datasets as dataframes

    Args:
        names (list): List of filenames names minus the json suffix.

    Returns:
        DefaultDict[str, pd.DataFrame]: Keys as names, values as the dataset
    """
    # Load datasets from json as dict
    datasets = load_from_json(names=names)

    # Process the names so that they are snake case
    var_names = [re.sub(pattern=r"-", repl="_", string=name) for name in names]

    # Name the datasets
    named_datasets = zip(var_names, datasets)

    # Find the evidence if available
    for var_name, dataset in named_datasets:
        if var_name == "evidence":
            df = DataFrame.from_dict(dataset, orient="index")
            df.columns = ["evidence_text"]
            locals()[var_name] = df

    # Parse each dataset into a DataFrame then join evidence if available
    datasets_df = list()
    for var_name, dataset in zip(var_names, datasets):
        if var_name == "evidence":
            continue
        df = DataFrame.from_dict(dataset, orient="index")
        if "evidence" in locals().keys():
            df = pd.merge(
                left=df.explode(column="evidences"),
                right=locals()["evidence"],
                how="left",
                left_on="evidences",
                right_index=True
            )
        df = df.reset_index(names=["claim"]) \
            .set_index(keys=["claim", "claim_text", "claim_label", "evidences"])
        datasets_df.append(df)

    return datasets_df
