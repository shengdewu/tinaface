import torch
import os
from vedacore.misc import load_weights
from vedadet.models import build_detector
from .post.box_bridge import BoxPost


class ScriptMode:
    def __init__(self, model_cfg, checkpoint, script_path):
        model = build_detector(model_cfg)
        load_weights(model, checkpoint, map_location='cpu')

        os.makedirs(script_path, exist_ok=True)
        model.eval()
        model = torch.jit.trace(model, torch.randn(1, 3, 640, 640))
        # model = torch.jit.script(model)
        torch.jit.save(model, os.path.join(script_path, 'tinaface.pt'))

        self.post = BoxPost()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(os.path.join(script_path, 'tinaface.pt')).to(self.device)
        return

    def __call__(self, data):
        feats = self.model(data['img'].to(self.device))

        result = self.post.simple_infer(feats, data['img_metas'].data[0])
        return result

