"""
Text normalisation tasks
"""
from dataclasses import dataclass
import re
from textacy.preprocessing import remove, normalize, make_pipeline
from functools import partial

@dataclass
class TokenRule:
    pattern: str
    repl: str

def repl_special_tokens(text:str) -> str:
    """
    Normalizes text by applying a set of pre-defined rules
    """
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
        ),
        TokenRule(
            pattern=r"\s*(%)",
            repl=" percent"
        )
    ]

    new_text = text
    for rule in rules:
        new_text = re.sub(
            pattern=rule.pattern,
            repl=rule.repl,
            string=new_text,
            flags=re.IGNORECASE
        )
    return new_text


def lower_case(text:str):
    """
    Converts all text into lower case.
    """
    return text.lower()


"""
Creates a normalizing pipeline by using the textacy framework
"""
normalize_pipeline =  make_pipeline(
    remove.brackets,
    remove.punctuation,
    remove.accents,
    normalize.whitespace,
    normalize.hyphenated_words,
    normalize.quotation_marks,
    normalize.unicode,
    partial(normalize.repeating_chars, chars=".!?"),
    lower_case,
    repl_special_tokens
)

if __name__ == "__main__":
    print(repl_special_tokens('In 1946 PDO switched to a cool phase'))
    pass