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

def process_sentence(text:str, nlp, bert_tokens:bool=False) -> str:
    """
    Preprocessor that:
    - lower case
    - lemmatise
    - remove stop words
    - remove negation

    Args:
        text (str): Text to transform.
        nlp (spacy model): A spacy nlp model.
        bert_tokens (bool): Whether to include BERT tokens.

    Returns:
        str: Transformed text
    """
    # If BERT tokens are required, make sure`sentencizer` is in pipeline
    # if bert_tokens and "sentencizer" not in nlp.pipe_names:
    #     nlp.add_pipe(factory_name="sentencizer", last=True)

    doc = nlp(text.lower())
    new_tokens = []
    for token in doc:
        # Exclude stop words
        if token.is_stop:
            continue

        # Exclude if token is a negator
        if token.dep_ == "neg":
            continue

        # Add the lemma of the token text
        new_tokens.append(token.lemma_)

        # Add whitespace if present
        if token.whitespace_:
            new_tokens.append(token.whitespace_)

        # Add the BERT [SEP] token if required
        if bert_tokens and token.is_sent_end:
            new_tokens += [" ", "[SEP]", " "]

    # Attach BERT [CLS] token if required
    if bert_tokens:
        new_tokens = ["[CLS]", " "] + new_tokens

    new_text = "".join(new_tokens)
    return new_text

    # Keep below code, it's a bit flawed but seems to give better
    # Similarity scores for some reason...
    # tokens = [
    #     token.lemma_
    #     for token in doc
    #     if not token.is_stop or not token.dep_ == "neg"
    # ]
    # return " ".join(tokens)