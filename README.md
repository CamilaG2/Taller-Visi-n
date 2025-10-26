# Taller de Visión por Computador – Inferencia CIFAR (VGG16 / ResNet50)

Proyecto académico para entrenamiento y despliegue de modelos de clasificación de imágenes mediante Transfer Learning y FastAPI.

## Ejecución de la API

```bash
cd src
uvicorn api_inferencia:app --reload
```

Luego abre en el navegador:
http://127.0.0.1:8000/docs

## Estructura

modelos/    → pesos entrenados (.pth)
src/        → código fuente (API e inferencia)
informe/    → informe LaTeX

## Autora

María Camila García Ramírez
Universidad del Rosario — Curso Introducción a la Visión por Computador (2025-2)