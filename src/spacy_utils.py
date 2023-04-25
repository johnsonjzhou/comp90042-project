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
            doc_text = \
                re.sub(pattern=rule.pattern, repl=rule.repl, string=doc_text)
        new_doc = self.nlp.make_doc(doc_text)
        return new_doc


@Language.factory("repl_special_token")
def repl_special_token(nlp, name):
    rules = [
        TokenRule(
            pattern=r"(CO\s*2)",
            repl="carbon dioxide"
        ),
        TokenRule(
            pattern=r"(PDO)",
            repl="pacific decadal oscillation"
        )
    ]
    return ReplaceSpecialToken(nlp=nlp, rules=rules)