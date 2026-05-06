from transformers import VisionEncoderDecoderModel, AutoTokenizer


def build_model():
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    # 🔥 CRITICAL FIXES (THIS SOLVES YOUR ERROR)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id

    model.config.eos_token_id = tokenizer.eos_token_id

    return model