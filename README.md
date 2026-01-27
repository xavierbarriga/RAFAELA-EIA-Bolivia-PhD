# RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido

> **Tesis Doctoral en Ciencias Exactas y Tecnología** > **Autor:** Ing. Xavier Eduardo Barriga Sinisterra  
> **Año:** 2026

![Status](https://img.shields.io/badge/Status-Doctoral_Thesis-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![AI Architecture](https://img.shields.io/badge/Architecture-Neuro_Symbolic-green)

## 📄 Resumen Ejecutivo

**RAFAELA** es una arquitectura de Inteligencia Artificial Híbrida (Neuro-Simbólica) diseñada para estandarizar y optimizar la revisión técnica de Estudios de Evaluación de Impacto Ambiental (EEIA) en el sector energético de Bolivia.

A diferencia de los modelos de lenguaje genéricos (LLMs), RAFAELA integra:
1.  **Modelo BERT (Fine-Tuned):** Para la comprensión semántica del lenguaje técnico boliviano.
2.  **RAG (Retrieval-Augmented Generation):** Un motor de recuperación basado en un *Gold Standard* de 502 observaciones validadas por expertos.
3.  **Ontología & Reglas:** Una capa lógica que asegura la coherencia normativa con la Ley 1333 y el D.S. 3549.

## 🧠 Arquitectura del Sistema

El sistema opera en tres fases cognitivas:

1.  **Fase de Percepción (Neural):** * El modelo `RAFAELA` clasifica la observación en 7 dimensiones taxonómicas (Biótico, Social, Físico, Deficiencia Técnica, etc.) y sugiere la normativa aplicable.
2.  **Fase de Recuperación (RAG):**
    * El sistema vectoriza la observación y busca los "Vecinos Más Cercanos" (Nearest Neighbors) en el *Gold Standard* para encontrar precedentes técnicos validados.
3.  **Fase de Generación (Simbólica):**
    * Se ensambla una propuesta de redacción técnica que combina el hallazgo del evaluador con el estilo formal y la fundamentación jurídica recuperada.

## 📂 Estructura del Repositorio

* `/src`: Código fuente del sistema (Entrenamiento, Inferencia y Auditoría).
    * `1_train_rafaela.py`: Script de entrenamiento del modelo BERT Multi-Task.
    * `2_inference_rafaela.py`: Pipeline de generación del *Silver Standard* (10k registros).
    * `3_audit_rafaela.py`: Módulo de generación de evidencia y gráficos.
* `/data`: Datasets utilizados (Muestras).
    * `GOLD_STANDARD_TRAIN.csv`: Corpus lingüístico anotado manualmente (502 registros).
* `/results`: Evidencia de validación del Hito 4 (Gráficos sectoriales y tablas de casos críticos).

## 🚀 Instalación y Reproducibilidad

Este proyecto requiere un entorno con soporte para GPU (recomendado).

```bash
# Clonar el repositorio
git clone [https://github.com/xavierbarriga/RAFAELA-EIA-Bolivia-PhD.git](https://github.com/xavierbarriga/RAFAELA-EIA-Bolivia-PhD.git)

# Instalar dependencias
pip install torch transformers pandas scikit-learn safetensors
