from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    # get_linear_schedule_with_warmup,
)

all_tokenizers = {"T5": lambda: T5Tokenizer.from_pretrained("t5-small")}

all_models = {"T5": lambda: T5ForConditionalGeneration.from_pretrained("t5-small")}
