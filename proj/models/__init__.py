from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    # get_linear_schedule_with_warmup,
)
from proj.constants import DISTILBERT_BIGRAM_TOKENIZER
from .lstm import lstmAttention, newsLSTM

all_tokenizers = {
    "T5": lambda: T5Tokenizer.from_pretrained("t5-small"),
    "distilBert": lambda: DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    ),
    "distilBertBigram": lambda: DistilBertTokenizer.from_pretrained(DISTILBERT_BIGRAM_TOKENIZER)
}


def getDistilBertBigram(*args):
    model = all_models["distilBert"](64)
    tokenizer = all_tokenizers['distilBertBigram']()
    model.resize_token_embeddings(len(tokenizer))
    return model


all_models = {
    "T5": lambda _: T5ForConditionalGeneration.from_pretrained(
        "t5-small", num_labels=10
    ),
    "distilBert": lambda _: DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=10
    ),
    "distilBertBigram": getDistilBertBigram,
    "lstm": lambda bs: newsLSTM(bs),
    "lstmAttention": lambda bs: lstmAttention(bs),
    "lstmAttentionBigram": lambda bs: lstmAttention(bs, useBigram=True),
    "lstmBigram": lambda bs: newsLSTM(bs, useBigram=True)
}
