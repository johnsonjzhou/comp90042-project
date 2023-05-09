"""
BERT based cross encoder classifier
"""

import torch
from torch.nn import Module, Linear, LogSoftmax, Dropout
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from src.torch_utils import get_torch_device
from typing import List, Tuple
from src.normalize import normalize_pipeline

class BertCrossEncoderClassifier(Module):

    def __init__(
            self,
            pretrained_name:str,
            n_classes:int=2,
            device=None,
            **kwargs
        ) -> None:
        super(BertCrossEncoderClassifier, self).__init__(**kwargs)
        self.device = device if device is not None else get_torch_device()

        # Use a pretrained tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)

        # Use a pretrained model
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.bert.to(device=self.device)

        # Classification head
        self.linear = Linear(768, n_classes, device=self.device)
        self.log_softmax = LogSoftmax(dim=1)
        return

    def forward(
        self,
        texts:List[List[str]],
        normalize_text:bool=True,
        max_length:int=128,
        dropout:float=None
    ) -> Tuple[torch.Tensor]:
        """
        Usage:
            For training, use the `logits` then `CrossEntropyLoss`.
            For inference, use the `output` which are the probabilities
                from `LogSoftmax` after applying `torch.exp`.

        Args:
            texts (List[List[str]]): [[claim_text, evidence_text]] of
                shape (n_batch, 2).
            normalize_text (bool): Whether to apply text normalization.
            max_length (int, optional): Padding length. Defaults to 128.
            dropout (float, optional): Apply dropout to the embeddings
                before the linear classification layer. Defaults to None.

        Returns:
            Tuple[torch.Tensor]:
                - output: Output from LogSoftmax.
                - logits: Logits from the linear classification layer before
                    LogSoftmax is applied.
                - seq: Pooler output from Bert model.
        """
        # Normalize if required
        if normalize_text:
            texts = [
                [normalize_pipeline(text) for text in text_pair]
                for text_pair in texts
            ]

        # Tokenize
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True
        ).to(device=self.device)

        # Apply Bert Embedder
        seq = self.bert(**tokens, return_dict=True).pooler_output

        # Apply dropout if desired
        if dropout is not None:
            dropout_fn = Dropout(p=dropout)
            seq = dropout_fn(seq)

        # Apply classification head
        logits = self.linear(seq)

        # Apply output
        output = self.log_softmax(logits)
        output = torch.exp(output)

        return output, logits, seq


class RobertaLargeCrossEncoderClassifier(Module):

    def __init__(
            self,
            pretrained_name:str,
            n_classes:int=2,
            device=None,
            **kwargs
        ) -> None:
        super(RobertaLargeCrossEncoderClassifier, self).__init__(**kwargs)
        self.device = device if device is not None else get_torch_device()

        # Use a pretrained tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)

        # Use a pretrained model
        self.bert = RobertaModel.from_pretrained(pretrained_name)
        self.bert.to(device=self.device)

        # Classification head
        self.linear = Linear(1024, n_classes, device=self.device)
        self.log_softmax = LogSoftmax(dim=1)
        return

    def forward(
        self,
        texts:List[List[str]],
        normalize_text:bool=True,
        max_length:int=128,
        dropout:float=None
    ) -> Tuple[torch.Tensor]:
        """
        Usage:
            For training, use the `logits` then `CrossEntropyLoss`.
            For inference, use the `output` which are the probabilities
                from `LogSoftmax` after applying `torch.exp`.

        Args:
            texts (List[List[str]]): [[claim_text, evidence_text]] of
                shape (n_batch, 2).
            normalize_text (bool): Whether to apply text normalization.
            max_length (int, optional): Padding length. Defaults to 128.
            dropout (float, optional): Apply dropout to the embeddings
                before the linear classification layer. Defaults to None.

        Returns:
            Tuple[torch.Tensor]:
                - output: Output from LogSoftmax.
                - logits: Logits from the linear classification layer before
                    LogSoftmax is applied.
                - seq: Pooler output from Bert model.
        """
        # Normalize if required
        if normalize_text:
            texts = [
                [normalize_pipeline(text) for text in text_pair]
                for text_pair in texts
            ]

        # Tokenize
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True
        ).to(device=self.device)

        # Apply Bert Embedder
        seq = self.bert(**tokens, return_dict=True).pooler_output

        # Apply dropout if desired
        if dropout is not None:
            dropout_fn = Dropout(p=dropout)
            seq = dropout_fn(seq)

        # Apply classification head
        logits = self.linear(seq)

        # Apply output
        output = self.log_softmax(logits)
        output = torch.exp(output)

        return output, logits, seq


def test():
    texts = [
        ["THE CO2 brown fox", "jumps over the lazy dog"],
        ["hello world", "foo bar"]
    ]
    model = RobertaLargeCrossEncoderClassifier(
        pretrained_name="roberta-large-mnli",
        n_classes=2
    )
    output, logits, seq = model(texts, max_length=512)

    for t, o, l, s in zip(texts, output, logits, seq):
        print(t, o, l)
    return

if __name__ == "__main__":
    test()
    pass