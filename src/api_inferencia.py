# src/api_inferencia.py
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Literal

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

import torch

# --- Utilidades de inferencia (de tu módulo) ---
from inferencia_modelos import load_model, predict_image, _dataset_classes

APP_TITLE = "API de Inferencia CIFAR (VGG16 / ResNet50)"
app = FastAPI(title=APP_TITLE)

# ======= RUTAS ABSOLUTAS (EDITA AQUÍ SI ES NECESARIO) =======
ABS_WEIGHTS_VGG16    = r"D:\Mis Documentos\Documentos\Rosario\2025-2\Vision\TallerVC\modelos\vgg16_cifar10.pth"
ABS_WEIGHTS_RESNET50 = r"D:\Mis Documentos\Documentos\Rosario\2025-2\Vision\TallerVC\modelos\resnet50_cifar100.pth"

# (Opcional) archivos de clases (uno por línea). Si es None, se usa helper interno / fallback.
CLASSLISTS: Dict[str, Optional[str]] = {
    "cifar10":  None,
    "cifar100": None,  # p.ej.: r"C:\...\cifar100_classes.txt"
}
# ============================================================

# --- Lista oficial de clases de CIFAR-100 (fallback si no hay .txt ni helper) ---
CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
]

# --- Registro de modelos disponibles ---
MODELS_REGISTRY: Dict[str, Dict] = {
    "vgg16": {
        "model_name":  "vgg16",
        "num_classes": 10,
        "weights":     ABS_WEIGHTS_VGG16,
        "dataset":     "cifar10",
        "img_size":    224,
    },
    "resnet50": {
        "model_name":  "resnet50",
        "num_classes": 100,
        "weights":     ABS_WEIGHTS_RESNET50,
        "dataset":     "cifar100",
        "img_size":    224,
    },
}

# Modelo por defecto si no se especifica
DEFAULT_MODEL_KEY = "vgg16"

# Cache de modelos cargados: key -> (model, transform, class_names)
_loaded: Dict[str, Tuple[torch.nn.Module, object, Optional[List[str]]]] = {}


def _load_class_names(dataset_name: Optional[str]) -> Optional[List[str]]:
    """Obtiene nombres de clases desde txt opcional, helper interno, o fallback."""
    if not dataset_name:
        return None

    ds = dataset_name.lower()

    # 1) .txt configurado
    txt_path = CLASSLISTS.get(ds)
    if txt_path:
        p = Path(txt_path)
        if not p.exists():
            raise HTTPException(status_code=500, detail=f"Archivo de clases no encontrado: {p}")
        with p.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
            if not names:
                raise HTTPException(status_code=500, detail=f"Archivo de clases vacío: {p}")
            return names

    # 2) helper interno (cubre CIFAR-10)
    names = _dataset_classes(ds)
    if names:
        return names

    # 3) fallback explícito para CIFAR-100
    if ds == "cifar100":
        return CIFAR100_CLASSES

    return None


def _ensure_loaded(model_key: str):
    """Carga perezosa del modelo solicitado (si no está en caché)."""
    if model_key not in MODELS_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Modelo no soportado: {model_key}")

    if model_key in _loaded:
        return

    cfg = MODELS_REGISTRY[model_key]
    weights_path = Path(cfg["weights"])
    if not weights_path.exists():
        raise HTTPException(status_code=500, detail=f"No existe el archivo de pesos: {weights_path}")

    model, transform, _ = load_model(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        weights_path=str(weights_path),
        img_size=cfg["img_size"],
    )
    class_names = _load_class_names(cfg["dataset"])
    _loaded[model_key] = (model, transform, class_names)


@app.get("/")
def root():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_KEY,
        "available_models": list(MODELS_REGISTRY.keys()),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Imagen a clasificar (jpg/png/bmp/webp)"),
    model: Optional[Literal["vgg16", "resnet50"]] = Query(None, description="Modelo a usar"),
    output: Literal["simple", "full"] = Query("simple", description="Formato de salida"),
    model_form: Optional[Literal["vgg16", "resnet50"]] = Form(None, description="Alternativa: enviar 'model' como form"),
):
    """
    Inferencia sobre una imagen:
      - Sube el archivo en 'file'
      - Elige el modelo vía query (?model=vgg16) o como campo de formulario 'model'
      - output: 'simple' -> {'prediction': '<label>'} | 'full' -> Top-1 + Top-K (según tu 'predict_image')
    """
    # Resolver modelo (prioriza QUERY sobre FORM y valida inconsistencia)
    # Resolver modelo (prioriza QUERY; si vienen ambos y difieren, solo avisa)
    model_q = model.lower() if model else None
    model_f = model_form.lower() if model_form else None

    model_key = model_q or model_f or DEFAULT_MODEL_KEY

    if model_q and model_f and (model_q != model_f):
        print(
            f"[WARN] 'model' por query ('{model_q}') difiere de 'model' por form ('{model_f}'). "
            f"Se prioriza el valor de la QUERY."
        )


    # Cargar modelo (si no está en caché)
    _ensure_loaded(model_key)
    mdl, transform, class_names = _loaded[model_key]

    # Guardar temporalmente el archivo subido
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            # Acepta igual, pero usa .jpg si no hay/extensión no estándar
            suffix = ".jpg" if not suffix else suffix

        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            content = await file.read()
            if not content:
                raise ValueError("El archivo está vacío.")
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer/guardar la imagen: {e}")

    # Ejecutar inferencia
    try:
        result = predict_image(
            image_path=tmp_path,
            model=mdl,
            transform=transform,
            class_names=class_names,  # ahora resnet50 tendrá nombres (CIFAR-100)
            output=output,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {e}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    # Ejecuta desde la raíz del proyecto:
    #   python -m uvicorn src.api_inferencia:app --reload
    uvicorn.run("src.api_inferencia:app", host="127.0.0.1", port=8000, reload=True)
