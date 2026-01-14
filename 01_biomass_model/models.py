from pathlib import Path
import torchvision.models as models
import torch.nn as nn

import torch

# -------------------- MODEL --------------------



def build_resnet(architecture, pretrained: bool = True):
    if architecture == "18":
        model = models.resnet18(pretrained=pretrained)
    elif architecture == "50":
        model = models.resnet50(pretrained=pretrained)
    elif architecture == "101":
        model = models.resnet101(pretrained=pretrained)
    else:
        raise Exception(f"Unsupported resnet architecture: {architecture}")
    # Replace final fc with one output (regression)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model



def load_weights_to_model(model: nn.Module, weights_path: Path, device: torch.device):
    """Attempt to load weights. Supports either saved state_dict or a full model object file.
    Returns model on `device`.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    data = torch.load(weights_path, map_location=device)
    # Heuristic: if dict with 'model_state_dict' key (checkpoint), load it;
    # if it's a state_dict (keys like 'fc.weight'), load directly; otherwise try load_state_dict block.
    if isinstance(data, dict):
        if "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            model.load_state_dict(state_dict)
        else:
            # could be a raw state_dict
            try:
                model.load_state_dict(data)
            except Exception:
                # maybe the user saved the whole model as dict with other keys; try common key names
                possible_keys = [k for k in data.keys() if isinstance(k, str) and k.endswith("state_dict")]
                if possible_keys:
                    model.load_state_dict(data[possible_keys[0]])
                else:
                    raise RuntimeError("Unable to interpret checkpoint structure. Please provide a state_dict or a checkpoint with key 'model_state_dict'.")
    else:
        # maybe they saved a full model object
        try:
            model = data.to(device)
            return model
        except Exception:
            raise RuntimeError("Loaded checkpoint is not a dict and cannot be placed on device as a model object.")

    return model.to(device)
