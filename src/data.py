"""
Data I/O functions
"""
import pathlib
import pandas as pd
import re
from typing import DefaultDict
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

def load_as_dataframe(names:list) -> DefaultDict[str, pd.DataFrame]:
    """
    Gets the json datasets as dataframes

    Args:
        names (list): List of filenames names minus the json suffix.

    Returns:
        DefaultDict[str, pd.DataFrame]: Keys as names, values as the dataset
    """
    DATA_PATH = pathlib.Path("data", "*")
    datasets = defaultdict(pd.DataFrame)
    for name in names:
        with open(DATA_PATH.with_name(name).with_suffix(".json")) as f:
            var_name = re.sub(pattern=r"-", repl="_", string=name)
            df = pd.read_json(f, orient="index")
            if name == "evidence":
                df.columns = ["text"]
            datasets[var_name] = df
            print(f"Loaded {var_name}")
    return datasets

def view_claim(claims:pd.DataFrame, evidence:pd.DataFrame) -> None:
    """
    Joins the claim and evidences and prints them for easier reading

    Args:
        claims (pd.DataFrame): The claims dataframe
        evidence (pd.DataFrame): The evidence dataframe
    """
    for index, data in claims.iterrows():
        print(COLORS.HEADER, f"{index} ==========", COLORS.END, "\n")
        for key in ["claim_text", "claim_label", "evidences"]:
            print(COLORS.OKBLUE, key, COLORS.END)
            print(data[key], "\n")
            if key == "evidences":
                evidences = evidence.loc[data[key]]
                for e_index, e_data in evidences.iterrows():
                    print(COLORS.OKGREEN, e_index, COLORS.END)
                    print(e_data["text"], "\n")
    return