from pathlib import Path
import torchvision.models as models
import torch.nn as nn

import torch


# -------------------- MODEL --------------------

def build_efficientnet(variant: str = "b0", pretrained: bool = True, n_outputs: int = 1):
    """
    variant: one of {"b0","b1","b2","b3","b4","b5","b6","b7","v2_s","v2_m","v2_l"}
    pretrained: uses ImageNet weights if available (torchvision)
    n_outputs: 1 for a point regression, 2 for a Gaussian (mean, log SD)
    """
    v = variant.lower()

    # --- EfficientNet v1 ---
    if v == "b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    elif v == "b1":
        model = models.efficientnet_b1(pretrained=pretrained)
    elif v == "b2":
        model = models.efficientnet_b2(pretrained=pretrained)
    elif v == "b3":
        model = models.efficientnet_b3(pretrained=pretrained)
    elif v == "b4":
        model = models.efficientnet_b4(pretrained=pretrained)
    elif v == "b5":
        model = models.efficientnet_b5(pretrained=pretrained)
    elif v == "b6":
        model = models.efficientnet_b6(pretrained=pretrained)
    elif v == "b7":
        model = models.efficientnet_b7(pretrained=pretrained)

    # --- EfficientNet v2 ---
    elif v == "v2_s":
        model = models.efficientnet_v2_s(pretrained=pretrained)
    elif v == "v2_m":
        model = models.efficientnet_v2_m(pretrained=pretrained)
    elif v == "v2_l":
        model = models.efficientnet_v2_l(pretrained=pretrained)

    else:
        raise Exception(f"Unsupported EfficientNet variant: {variant}")

    # Replace classifier head with a regression output of n_outputs values
    # torchvision EfficientNet has model.classifier = Sequential(Dropout, Linear)
    if isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),  # ← stronger dropout
            nn.Linear(in_features, n_outputs)
        )
    else:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, n_outputs)
        )
    return model


class ScaleInputNet(nn.Module):
    """EfficientNet whose head also receives physical scale as numbers (log L, log area).

    The image features go through the usual dropout; the scale inputs bypass it, are
    standardised with buffers set from the training data (saved in the state dict), and are
    concatenated before the final linear layer. Used for the reviewer comparison: a direct
    log-mass model that is told the specimen's size instead of having it built in as BF.
    """

    def __init__(self, variant: str = "v2_s", pretrained: bool = True, n_outputs: int = 2, n_scale: int = 2):
        super().__init__()
        self.backbone = build_efficientnet(variant, pretrained, n_outputs)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(p=0.6)
        self.head = nn.Linear(in_features + n_scale, n_outputs)
        self.register_buffer("scale_mean", torch.zeros(n_scale))
        self.register_buffer("scale_std", torch.ones(n_scale))

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        s = (scale - self.scale_mean) / self.scale_std
        return self.head(torch.cat([self.dropout(self.backbone(x)), s], dim=1))


def load_weights_to_model(model: nn.Module, weights_path: Path, device: torch.device):
    """Attempt to load weights. Supports either saved state_dict or a full model object file.
    Returns model on `device`.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    data = torch.load(weights_path, map_location=device)

    if isinstance(data, dict):
        if "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            model.load_state_dict(state_dict)
        else:
            try:
                model.load_state_dict(data)
            except Exception:
                possible_keys = [
                    k for k in data.keys()
                    if isinstance(k, str) and k.endswith("state_dict")
                ]
                if possible_keys:
                    model.load_state_dict(data[possible_keys[0]])
                else:
                    raise RuntimeError(
                        "Unable to interpret checkpoint structure. Please provide a state_dict "
                        "or a checkpoint with key 'model_state_dict'."
                    )
    else:
        # maybe they saved a full model object
        try:
            model = data.to(device)
            return model
        except Exception:
            raise RuntimeError(
                "Loaded checkpoint is not a dict and cannot be placed on device as a model object."
            )

    return model.to(device)


def load_gaussian_model(weights_path: Path, device: torch.device, variant: str = "v2_s") -> nn.Module:
    """Load a 2-output (mu, log sigma) checkpoint, as a ScaleInputNet if it was trained with
    scale inputs (its state dict then holds `scale_mean`), else as a plain EfficientNet."""
    data = torch.load(Path(weights_path), map_location=device)
    state = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
    if "scale_mean" in state:
        model = ScaleInputNet(variant=variant, pretrained=False, n_outputs=2)
    else:
        model = build_efficientnet(variant=variant, pretrained=False, n_outputs=2)
    model.load_state_dict(state)
    return model.to(device).eval()


def call_gaussian_model(model: nn.Module, images: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Scale-input models take (images, [log L, log area]); the others ignore the scales."""
    return model(images, scales) if isinstance(model, ScaleInputNet) else model(images)
