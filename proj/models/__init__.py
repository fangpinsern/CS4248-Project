import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    # get_linear_schedule_with_warmup,
)

from .lstm import lstmAttention, newsLSTM

all_tokenizers = {
    "T5": lambda: T5Tokenizer.from_pretrained("t5-small"),
    "distilBert": lambda: DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    ),
}

all_models = {
    "T5": lambda _: T5ForConditionalGeneration.from_pretrained(
        "t5-small", num_labels=10
    ),
    "distilBert": lambda _: DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=10
    ),
    "lstm": lambda bs: newsLSTM(bs),
    "lstmAttention": lambda bs: lstmAttention(bs),
}
