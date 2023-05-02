import json
import os
import torch
from typing import Any


def load_model(id: str):
    """
    Load a model by ID.
    """

    model: torch.nn.Module = torch.load(f"models/{id}/model.pt")
    model.eval()

    return model


def save_model(model: torch.nn.Module, id: str, metadata: Any = {}):
    """
    Save a model by ID, and associated optional metadata.
    """

    models_dir = "models"

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    model_dir = f"{models_dir}/{id}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    torch.save(model, f"{model_dir}/model.pt")

    with open(f"{model_dir}/model.json", "w") as f:
        json.dump(metadata, f, indent=None)
