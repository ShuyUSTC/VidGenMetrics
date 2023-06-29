"""
Code for Video version of CLIPScore (https://arxiv.org/abs/2104.08718)

By Yan Shu
shuy.ustc@gmail.com
"""
from transformers import CLIPProcessor, CLIPModel

import torch
import numpy as np


class FrameAcc:
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
            return_type: return type. support: pt (torch.Tensor), np(numpy.Ndarray), float (float)
        """
        assert return_type in self.RETURN_TYPE, f'Got return type: {return_type}, but only support f{self.RETURN_TYPE}'
        self.return_type = return_type
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version).to(dtype=torch.float16, device=device)
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
    def __call__(self, texts, frames, w=2.5):
        """
        Compute CLIP score between frames and texts

        Args:
            texts (list[str] | str):
            frames (list[Image | np.Ndarray] | Image | np.Ndarray):

        Returns:
            CLIP similarity(ies)

        """
        mini_bsz = self.mini_bsz if self.mini_bsz > 0 else len(frames)
        mini_bsz = min(mini_bsz, len(frames))
        clip_score_per_frame = []
        for b_idx in range((len(frames) + mini_bsz - 1) // mini_bsz):
            frame_batch = frames[b_idx * mini_bsz: b_idx * mini_bsz + mini_bsz]
            inputs = self.processor(text=texts, images=frame_batch, truncation=True, max_length=self.max_length,
                                    return_overflowing_tokens=False, padding="max_length",
                                    return_tensors="pt")
            inputs = inputs.to(device=self.device)
            for key in inputs:
                if inputs[key].dtype == torch.float:
                    inputs[key] = inputs[key].to(torch.float16)
            outputs = self.model(**inputs)
            clip_score_per_frame.append(outputs.logits_per_image)
        clip_score_per_frame = torch.cat(clip_score_per_frame)
        clip_score = w * clip_score_per_frame.mean(dim=0).cpu()
        if self.return_type == 'pt':
            return clip_score
        elif self.return_type == 'np':
            return np.array(clip_score)
        elif self.return_type == 'float':
            return [float(sim) for sim in clip_score] if len(clip_score) > 1 else float(clip_score)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    import glob
    from PIL import Image
    prompt = 'a silver jeep driving down a curvy road in the countryside.'
    images = glob.glob('./examples/*.png')
    images.sort()
    images = [Image.open(img) for img in images]
    frame_acc = FrameAcc()
    print(frame_acc(prompt, images))
