# skin-diseases-dl-project

Proyecto final de Deep Learning orientado a la clasificación de enfermedades dermatológicas a partir de imágenes clínicas.

El desarrollo principal se concentra en un notebook de investigación y experimentación que documenta las cinco etapas del proyecto:
- Línea base con redes neuronales.
- Arquitectura profunda especializada.
- Modelos preentrenados y Transformers.
- Componente generativo para balanceo y análisis.
- Fine-tuning, optimización y despliegue.

Los artefactos de despliegue no se entrenan desde la app. Se generan desde el notebook, se exportan manualmente y luego se cargan al bucket de Google Drive utilizado por la demo de Streamlit.

## Qué busca el proyecto

El objetivo es evaluar distintas estrategias de modelado y despliegue para clasificación multiclase de imágenes de piel, comparando enfoques progresivamente más expresivos y más eficientes:
- un baseline neuronal simple,
- una CNN profunda especializada,
- modelos preentrenados basados en Transformers,
- técnicas generativas para exploración y aumento de datos,
- y estrategias de fine-tuning y optimización para inferencia.

## Dataset

Se trabaja con el dataset *Skin Diseases Cancer Comprehensive Dataset* de Kaggle, un conjunto de imágenes organizadas por clase.

Características generales:
- Tipo de dato: imágenes dermatológicas clasificadas por etiqueta.
- Estructura: carpetas por clase.
- Uso principal: clasificación multiclase.
- Análisis realizado: distribución de clases, metadatos de imagen, tamaños, dimensiones, relación de aspecto y revisión de calidad básica.

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- PyTorch
- Transformers y Hugging Face
- PEFT / LoRA
- Streamlit
- Google Drive como repositorio manual de artefactos para la demo
- Jupyter Notebook para el flujo principal de experimentación

## Flujo del proyecto

1. El notebook principal entrena y evalúa los modelos por etapa.
2. Se generan artefactos de despliegue con el modelo final y los metadatos asociados.
3. Los artefactos se suben manualmente a Google Drive.
4. La app de Streamlit descarga esos artefactos y ejecuta inferencia sobre imágenes cargadas por el usuario.

## Estructura relevante

- [notebooks/skin-diseases-dl-udb.ipynb](notebooks/skin-diseases-dl-udb.ipynb): notebook principal con todo el desarrollo.
- [build/skin-classifier-demo/app.py](build/skin-classifier-demo/app.py): demo de inferencia en Streamlit.
- [build/skin-classifier-demo/requirements.txt](build/skin-classifier-demo/requirements.txt): dependencias de la demo.
- [notebooks/skin-diseases-etapa-uno.ipynb](notebooks/skin-diseases-etapa-uno.ipynb): notebook recortado de Etapa 1.
- [notebooks/skin-diseases-etapa-dos.ipynb](notebooks/skin-diseases-etapa-dos.ipynb): notebook recortado de Etapas 1 y 2.
- [notebooks/skin-diseases-etapa-tres.ipynb](notebooks/skin-diseases-etapa-tres.ipynb): notebook recortado de Etapas 1, 2 y 3.
- [notebooks/skin-diseases-etapa-cuatro.ipynb](notebooks/skin-diseases-etapa-cuatro.ipynb): notebook recortado de Etapas 1 a 4.

## Demo

La demo web consume los artefactos publicados en Google Drive y permite hacer predicción sobre una imagen cargada por el usuario.

Enlaces:
- Demo: https://skin-diseases-dl-project-zpa9aq5ussumteuapppbtvc.streamlit.app/
- Bucket (Google Drive): https://drive.google.com/drive/folders/1ScflbX21RFYYUHOjy7_lOSacWRYRz5uR
- Video explicativo del flujo: 

## Artefactos esperados en Drive

La demo busca estos archivos en el bucket:
- `skin_disease_efficientnet_ft5.keras`
- `model_metadata.json`
- `class_labels.json`

## Nota operativa

Los archivos de despliegue se actualizan manualmente desde el notebook. La app de Streamlit solo descarga y consume los artefactos ya generados.