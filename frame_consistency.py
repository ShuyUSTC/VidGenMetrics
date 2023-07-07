"""
Code for frame consistency
By Yan Shu
shuy.ustc@gmail.com
"""
import numpy as np
import torch
from einops import rearrange
from torch.nn import CosineSimilarity
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class FrameConsistency:
    RETURN_TYPE = [
        'pt',
        'np',
        'float',
    ]

    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", mini_bsz=64, return_type='pt'):
        """Measure frame consistency by computing the CLIP consine similarity of consecutive frames
        Args:
            version: clip model version
            device: device
            mini_bsz: mini batch size to process frames
            return_type: return type. support: pt (torch.Tensor), np(numpy.ndarray), float (float)
        """
        assert return_type in self.RETURN_TYPE, f'Got return type: {return_type}, but only support f{self.RETURN_TYPE}'
        self.return_type = return_type
        self.processor = CLIPImageProcessor.from_pretrained(version)
        self.model = CLIPVisionModelWithProjection.from_pretrained(version).to(device)
        self.cosine = CosineSimilarity(1).to(device)
        self.device = device
        self.mini_bsz = mini_bsz
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def __call__(self, frames, step=1):
        """
        Compute frame consistency of consecutive frames in `frames`

        Args:
            frames (list[Image | np.ndarray] | Image | np.ndarray):
            step: frame step to compute consistency

        Returns:
            frame consistency

        """
        num_frames = len(frames)
        assert num_frames > 1, 'Need at least 2 frames to compute frame consistency'
        assert num_frames > step, ''
        frame_embeds = []
        batch_size = self.mini_bsz if self.mini_bsz > 0 else len(frames)
        for b_idx in range(0, num_frames, batch_size):
            inputs = self.processor(frames[b_idx: b_idx + batch_size], return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            frame_embeds.append(outputs.image_embeds)
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

        if self.return_type == 'pt':
            return consist
        elif self.return_type == 'np':
            return np.array(consist)
        elif self.return_type == 'float':
            return [float(sim) for sim in consist] if len(consist) > 1 else float(consist)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    import glob
    from PIL import Image
    images = glob.glob('./examples/*.png')
    images.sort()
    images = [Image.open(img) for img in images]
    frame_consistency = FrameConsistency()
    print(frame_consistency(images))  # Also support List of np.ndarray with dtype as np.uint8
