from transformers import (
    CLIPFeatureExtractor,
    CLIPTokenizer,
    FlaxCLIPTextModel,
    FlaxT5EncoderModel,
    T5Tokenizer,
)
import jax
import jax.numpy as jnp
import numpy as np
import flax
import json

def load_models():
    weight_dtype = jnp.float32

    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    t5_encoder_model, t5_encoder_params = FlaxT5EncoderModel.from_pretrained(
        "google/flan-t5-base", dtype=weight_dtype, _do_init=False
    )

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder_model, clip_text_encoder_params = FlaxT5EncoderModel.from_pretrained(
        "openai/clip-vit-large-patch14", dtype=weight_dtype, _do_init=False
    )   
