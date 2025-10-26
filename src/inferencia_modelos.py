from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

__all__ = [
    "imagenet_transforms",
    "_dataset_classes",
    "_replace_last_layer",
    "load_model",
    "predict_image",
]

# --- Normalización / tamaño consistente con el entrenamiento ---
def imagenet_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# --- Cargar clases por dataset (opcional) ---
def _dataset_classes(dataset_name: Optional[str]) -> Optional[List[str]]:
    if not dataset_name:
        return None
    d = dataset_name.lower()
    if d == "cifar10":
        return ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    if d == "cifar100":
        # Si quieres nombres legibles, pásalos externamente (lista de 100 strings)
        return None
    return None

# --- Reemplazo de última capa según modelo ---
def _replace_last_layer(model_name: str, net: nn.Module, num_classes: int) -> nn.Module:
    m = model_name.lower()
    if m == "vgg16":
        in_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features, num_classes)
    elif m == "resnet50":
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("model_name debe ser 'vgg16' o 'resnet50'.")
    return net

def load_model(
    model_name: str,
    num_classes: int,
    weights_path: Union[str, Path],
    device: Optional[torch.device] = None,
    img_size: int = 224,
) -> Tuple[nn.Module, transforms.Compose, int]:
    """
    Carga arquitectura + reemplaza la última capa + aplica pesos entrenados.
    Devuelve: (modelo_en_device, transform, img_size)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model_name.lower()

    if m == "vgg16":
        net = models.vgg16(weights=None)
    elif m == "resnet50":
        net = models.resnet50(weights=None)
    else:
        raise ValueError("model_name debe ser 'vgg16' o 'resnet50'.")

    net = _replace_last_layer(m, net, num_classes)
    state = torch.load(str(weights_path), map_location="cpu")
    net.load_state_dict(state, strict=True)
    net.eval().to(device)

    tfm = imagenet_transforms(img_size=img_size)
    return net, tfm, img_size

def _resolve_class_name(i: int, class_names: Optional[List[str]]) -> str:
    if class_names and 0 <= i < len(class_names):
        return class_names[i]
    return str(i)

@torch.inference_mode()
def predict_image(
    image_path: Union[str, Path],
    model: nn.Module,
    transform: transforms.Compose,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    topk: int = 5,
    output: str = "full",  # "full" (detallado) o "simple"
) -> Dict:
    """
    Ejecuta inferencia sobre una imagen.

    output:
      - "full" → Top-1 + Top-K (detallado)
      - "simple" → {"prediction": "<etiqueta>"}
    """
    device = device or next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    k = min(topk, probs.shape[1])
    conf, idx = probs.topk(k=k, dim=1)

    conf = conf[0].detach().cpu().tolist()
    idx = idx[0].detach().cpu().tolist()

    top1_idx = idx[0]
    top1_name = _resolve_class_name(top1_idx, class_names)

    if output == "simple":
        return {"prediction": top1_name}

    pred_top1 = {"class_index": top1_idx, "class_name": top1_name, "prob": float(conf[0])}
    topk_list = [
        {"class_index": i, "class_name": _resolve_class_name(i, class_names), "prob": float(p)}
        for i, p in zip(idx, conf)
    ]
    return {"prediction": pred_top1, "topk": topk_list}
