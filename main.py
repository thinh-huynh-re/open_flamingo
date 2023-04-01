import torch
from open_flamingo import create_model_and_transforms
from pathlib import Path
from PIL import Image

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=Path(
        "cache/models--aleksickx--llama-7b-hf/snapshots/d7d132438caf5e95800f35dfc46cf82c2be9b365"
    ),
    tokenizer_path=Path(
        "cache/models--aleksickx--llama-7b-hf/snapshots/d7d132438caf5e95800f35dfc46cf82c2be9b365"
    ),
    cross_attn_every_n_layers=4,
    use_local_files=True,
)
print('Initialized models')

device = torch.device("cpu")

checkpoint_path = "cache/models--openflamingo--OpenFlamingo-9B/snapshots/b5cd34cb6c90775b262837b6a80a6a47123b4571/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path), strict=False)
print('Loaded weights')
model.to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Num parameters', "{:,}".format(pytorch_total_params))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
"""
Step 1: Load images
"""
demo_image_one = Image.open("samples/000000039769.jpg")
demo_image_two = Image.open("samples/000000028137.jpg")
query_image = Image.open("samples/000000028352.jpg")


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
vision_x = [
    image_processor(demo_image_one).unsqueeze(0),
    image_processor(demo_image_two).unsqueeze(0),
    image_processor(query_image).unsqueeze(0),
]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
lang_x = tokenizer(
    [
        "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"
    ],
    return_tensors="pt",
)

vision_x = vision_x.to(device)
lang_x = lang_x.to(device)

"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
