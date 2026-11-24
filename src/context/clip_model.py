"""CLIP image and text feature extraction."""

import torch
import clip
from PIL import Image


class CLIPModel:
    def __init__(self, model_name, device, scene_prompts):
        self.device = device
        self.model, self.preprocess = clip.load(
            model_name,
            device=device,
        )
        self.model.eval()

        self.scene_prompts = scene_prompts

        with torch.inference_mode():
            tokens = clip.tokenize(scene_prompts).to(device)
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(
                dim=-1, keepdim=True
            )

    @torch.inference_mode()
    def encode_image(self, frame):
        image = Image.fromarray(frame[:, :, ::-1])
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        features = self.model.encode_image(image_tensor)
        return features / features.norm(dim=-1, keepdim=True)
