"""
Code for Video version of CLIPScore, including frame accuracy and frame consistency

By Yan Shu
shuy.ustc@gmail.com
"""

import numpy as np
import torch
from einops import rearrange
from torch.nn import CosineSimilarity
from transformers import CLIPProcessor, CLIPModel


class CLIPScore:
    RETURN_TYPE = [
        'pt',
        'np',
        'float',
    ]

    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", prefix="A photo depicts",
                 max_length=77, mini_bsz=64, return_type='pt'):
        """Compute CLIP score between video frames and corresponding prompt
        Args:
            version: clip model version
            device: device
            prefix: prefix added to the prompt
            max_length: max_length for text tokenizer
            mini_bsz: mini batch size to process frames
            return_type: return type. support: pt (torch.Tensor), np(numpy.ndarray), float (float)
        """
        assert return_type in self.RETURN_TYPE, f'Got return type: {return_type}, but only support f{self.RETURN_TYPE}'
        self.return_type = return_type
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version).to(dtype=torch.float16, device=device)
        self.cosine = CosineSimilarity(1).to(device)
        self.prefix = prefix
        self.device = device
        self.max_length = max_length
        self.mini_bsz = mini_bsz
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def __call__(self, texts, frames, step=1, w=1.):
        """
        Compute CLIP score between frames and texts

        Args:
            texts (list[str] | str):
            frames (list[Image | np.ndarray] | Image | np.ndarray):

        Returns:
            CLIP similarity(ies)

        """
        num_frames = len(frames)
        mini_bsz = self.mini_bsz if self.mini_bsz > 0 else num_frames
        mini_bsz = min(mini_bsz, num_frames)
        acc_per_frame = []
        frame_embeds = []
        for b_idx in range((num_frames + mini_bsz - 1) // mini_bsz):
            frame_batch = frames[b_idx * mini_bsz: b_idx * mini_bsz + mini_bsz]
            inputs = self.processor(text=texts, images=frame_batch, truncation=True, max_length=self.max_length,
                                    return_overflowing_tokens=False, padding="max_length",
                                    return_tensors="pt")
            inputs = inputs.to(device=self.device)
            for key in inputs:
                if inputs[key].dtype == torch.float:
                    inputs[key] = inputs[key].to(torch.float16)
            outputs = self.model(**inputs)
            acc_per_frame.append(outputs.logits_per_image)
            frame_embeds.append(outputs.image_embeds)
        acc_per_frame = torch.cat(acc_per_frame)
        acc = w * acc_per_frame.mean(dim=0).cpu()

        frame_embeds = torch.cat(frame_embeds)
        rest_frames = num_frames % step
        rest_embeds = frame_embeds[num_frames-rest_frames:] if rest_frames > 0 else None
        frame_embeds = rearrange(frame_embeds[:num_frames-rest_frames], '(t s) c -> s c t', s=step)
        if rest_frames > 0:
            rest_embeds = torch.cat((frame_embeds[:rest_frames], rest_embeds), dim=2)
            consist = self.cosine(rest_embeds[..., :-1], rest_embeds[..., 1:]).cpu()
            consist = consist.mean(dim=1)
            if frame_embeds.shape[2] > 1:
                consist_short = self.cosine(frame_embeds[rest_frames:, :, :-1], frame_embeds[rest_frames:, :, 1:]).cpu()
                consist = torch.cat((consist, consist_short.mean(dim=1)))
        else:
            consist = self.cosine(frame_embeds[..., :-1], frame_embeds[..., 1:]).cpu()
            consist = consist.mean(dim=1)
        consist = consist.mean()
        # consist = self.cosine(frame_embeds[:-1], frame_embeds[1:]).cpu()
        consist = consist.mean()

        if self.return_type == 'pt':
            return {'frame_accuracy': acc, 'frame_consistency': consist}
        elif self.return_type == 'np':
            return {'frame_accuracy': np.array(acc), 'frame_consistency': np.array(consist)}
        elif self.return_type == 'float':
            return {'frame_accuracy': [float(sim) for sim in acc] if len(acc) > 1 else float(acc),
                    'frame_consistency': [float(sim) for sim in consist] if len(consist) > 1 else float(consist)}
        else:
            raise NotImplementedError


if __name__ == '__main__':
    import glob
    from PIL import Image
    prompt = 'a silver jeep driving down a curvy road in the countryside.'
    images = glob.glob('./examples/*.png')
    images.sort()
    images = [Image.open(img) for img in images]
    score = CLIPScore()
    print(score(prompt, images))  # Also support List of np.ndarray with dtype as np.uint8
