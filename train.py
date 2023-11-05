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
from typing import Any
from flax import struct

dtype = Any


class FrozenModel(struct.PyTreeNode):
    """
    mimic the behaviour of train_state but this time for frozen params
    to make it passable to the jitted function
    """

    # use pytree_node=False to indicate an attribute should not be touched
    # by Jax transformations.
    model_obj: Any = struct.field(pytree_node=False)
    params: dict = struct.field(pytree_node=True)


def load_models(
    t5_path_or_hf_repo: str = "google/flan-t5-base",
    clip_path_or_hf_repo: str = "openai/clip-vit-large-patch14",
    frozen_dict_dtype: dtype = jnp.float32,
):
    t5_tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=t5_path_or_hf_repo
    )
    t5_encoder_model, t5_encoder_params = FlaxT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path=t5_path_or_hf_repo,
        dtype=frozen_dict_dtype,
        _do_init=False,
    )
    frozen_t5_encoder_model = FrozenModel(
        model_obj=t5_encoder_model,
        params=t5_encoder_params,
    )

    clip_tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path=clip_path_or_hf_repo
    )
    (
        clip_text_encoder_model,
        clip_text_encoder_params,
    ) = FlaxCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=clip_path_or_hf_repo,
        dtype=frozen_dict_dtype,
        _do_init=False,
    )
    frozen_clip_encoder_model = FrozenModel(
        model_obj=clip_text_encoder_model,
        params=clip_text_encoder_params,
    )

    return {
        "frozen_t5": frozen_t5_encoder_model,
        "frozen_clip": frozen_clip_encoder_model,
    }


frozen_models = load_models()

print()
