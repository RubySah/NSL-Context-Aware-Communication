"""CLIP image and text features for two-stage scene understanding."""

import torch
import clip
from PIL import Image


class CLIPModel:
    def __init__(self, model_name, device, scene_prompts, scene_objects, object_template):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Stage 1: one embedding per scene, the average of its prompts.
        self.scene_names = list(scene_prompts)
        self.scene_features = torch.stack(
            [self._mean_text_feature(prompts) for prompts in scene_prompts.values()]
        )

        # Stage 2: one embedding per object, grouped by scene.
        self.object_names = {scene: list(items) for scene, items in scene_objects.items()}
        self.object_features = {
            scene: self._text_features([object_template.format(item) for item in items])
            for scene, items in scene_objects.items()
        }

    # Normalized CLIP embeddings for a list of texts.
    @torch.inference_mode()
    def _text_features(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True)

    # Average several prompts into one normalized embedding.
    def _mean_text_feature(self, texts):
        mean = self._text_features(texts).mean(dim=0)
        return mean / mean.norm()

    # Normalized CLIP embedding of one BGR camera frame.
    @torch.inference_mode()
    def encode_image(self, frame):
        image = Image.fromarray(frame[:, :, ::-1])
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor).float()
        return features / features.norm(dim=-1, keepdim=True)

    # Probability of each scene for the image (sums to 1).
    def scene_probabilities(self, image_features):
        logits = 100.0 * image_features @ self.scene_features.T
        return logits.softmax(dim=-1)[0]

    # Best-matching objects from one scene's list, keeping only clear matches.
    def top_objects(self, image_features, scene, k, min_similarity):
        similarity = (image_features @ self.object_features[scene].T)[0]
        order = similarity.argsort(descending=True)[:k].tolist()
        return [
            self.object_names[scene][i]
            for i in order
            if similarity[i].item() >= min_similarity
        ]