"""
Additional utilities for enhancing spaCy functionality
"""
import spacy
from spacy.language import Language
import re
from dataclasses import dataclass
from typing import List

@dataclass
class TokenRule:
    pattern: str
    repl: str


class ReplaceSpecialToken(object):
    """
    Use as the first pipe. Replaces text based on RegEx rules defined by
    a list of `TokenRule` objects.
    """
    def __init__(self, nlp, rules:List[TokenRule]):
        self.nlp = nlp
        self.rules = rules
        return

    def __call__(self, doc):
        doc_text = doc.text
        for rule in self.rules:
            doc_text = re.sub(
                pattern=rule.pattern,
                repl=rule.repl,
                string=doc_text,
                flags=re.IGNORECASE
            )
        new_doc = self.nlp.make_doc(doc_text)
        return new_doc


@Language.factory("repl_special_token")
def repl_special_token(nlp, name):
    rules = [
        TokenRule(
            pattern=r"\s*(CO\s*2)",
            repl=" carbon dioxide"
        ),
        TokenRule(
            pattern=r"(PDO)",
            repl="pacific decadal oscillation"
        ),
        TokenRule(
            pattern=r"(CFC)",
            repl="chlorofluorocarbon"
        ),
        TokenRule(
            pattern=r"\s*(°\s*N)",
            repl=" degree north latitude"
        ),
        TokenRule(
            pattern=r"\s*(°\s*S)",
            repl=" degree south latitude"
        ),
        TokenRule(
            pattern=r"\s*(°\s*E)",
            repl=" degree east longitude"
        ),
        TokenRule(
            pattern=r"\s*(°\s*W)",
            repl=" degree west longitude"
        ),
        TokenRule(
            pattern=r"\s*(°\s*C)",
            repl=" degree celcius temperature"
        ),
        TokenRule(
            pattern=r"\s*(°\s*F)",
            repl=" degree fahrenheit temperature"
        )
    ]
    return ReplaceSpecialToken(nlp=nlp, rules=rules)

def process_sentence(text:str, nlp) -> str:
    """
    Preprocessor that:
    - lower case
    - lemmatise
    - remove stop words
    - remove negation

    Args:
        text (str): Text to transform.
        nlp (spacy model): A spacy nlp model.

    Returns:
        str: Transformed text
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop or not token.dep_ == "neg"
    ]
    return " ".join(tokens)