from dataclasses import dataclass, field

import numpy as np
import torch

from ...extras.constants import IGNORE_INDEX


def _numpify(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return x


def eval_logit_processor(logits, labels):
    if isinstance(logits, (list, tuple)):
        logits = logits[0] if logits[0].dim() == 3 else logits[1]
    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    score_dict: dict = field(default_factory=lambda: {"accuracy": []})

    def __call__(self, eval_preds, compute_result=True):
        preds, labels = _numpify(eval_preds.predictions), _numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[mask] == label[mask]) if mask.any() else 0.0)
        if compute_result:
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            self.score_dict = {"accuracy": []}
            return result
        return None
