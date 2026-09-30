"""Model construction, mirroring Phase 4's models/video_transformer.py."""

import torch
import torch.nn as nn


def build_videomae(num_classes, cache_dir):
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        cache_dir=cache_dir, num_labels=num_classes, ignore_mismatched_sizes=True,
    )
    return model


def build_mvit(num_classes, cache_dir):
    import torchvision
    torch.hub.set_dir(str(cache_dir))
    model = torchvision.models.video.mvit_v2_s(weights="KINETICS400_V1")
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(backbone, num_classes, cache_dir):
    if backbone == "videomae":
        return build_videomae(num_classes, cache_dir)
    elif backbone == "mvit":
        return build_mvit(num_classes, cache_dir)
    raise ValueError(f"Unknown backbone: {backbone!r}")


def forward_logits(backbone, model, clips):
    if backbone == "videomae":
        outputs = model(pixel_values=clips)
        return outputs.logits
    elif backbone == "mvit":
        clips_mvit = clips.permute(0, 2, 1, 3, 4)
        return model(clips_mvit)
    raise ValueError(f"Unknown backbone: {backbone!r}")
