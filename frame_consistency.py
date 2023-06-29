"""
Code for frame consistency
By Yan Shu
shuy.ustc@gmail.com
"""
import numpy as np
import torch
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
            return_type: return type. support: pt (torch.Tensor), np(numpy.Ndarray), float (float)
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
    def __call__(self, frames):
        """
        Compute frame consistency of consecutive frames in `frames`

        Args:
            frames (list[Image | np.Ndarray] | Image | np.Ndarray):

        Returns:
            frame consistency

        """
        assert len(frames) > 1, 'Need at least 2 frames to compute frame consistency'
        inputs = self.processor(frames, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        frame_embeds = outputs.image_embeds
        consist = self.cosine(frame_embeds[:-1], frame_embeds[1:]).cpu()
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
    print(frame_consistency(images))
