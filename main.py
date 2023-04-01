from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="aleksickx/llama-7b-hf",
    tokenizer_path="aleksickx/llama-7b-hf",
    cross_attn_every_n_layers=4,
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-9B", "checkpoint.pt", cache_dir="cache"
)
model.load_state_dict(torch.load(checkpoint_path), strict=False)
