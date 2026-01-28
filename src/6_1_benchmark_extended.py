# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE BENCHMARKING GENERATIVO EXTENDIDO (SLM Battle Arena)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 6.1 (Extended)

Descripción:
Ejecuta un torneo comparativo de alto nivel entre las arquitecturas SLM (Small Language Models)
más relevantes del estado del arte (SOTA) 2024-2025 para la redacción técnica en español.

MODELOS EVALUADOS:
1. Baseline: Reglas Deterministas (Control).
2. Seq2Seq: Google Flan-T5 (Legacy robusto).
3. Llama-Arch: TinyLlama 1.1B (Eficiencia pura).
4. Phi-Arch: Microsoft Phi-3 Mini (Razonamiento lógico superior).
5. Gemma-Arch: Google Gemma 2B (Arquitectura Gemini ligera).
6. Qwen-Arch: Qwen2-1.5B (Rendimiento multilingüe/español superior).

MECANISMO:
Utiliza carga secuencial con limpieza de VRAM/RAM (Garbage Collection) para permitir
la ejecución de múltiples modelos en hardware limitado.
"""

import pandas as pd
import json
import torch
import gc
import os
import sys
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "SILVER_STANDARD_FINAL_RAG.csv")
ONTOLOGY_FILE = os.path.join(BASE_DIR, "..", "rafaela_ontology.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "RESULTADOS_BENCHMARK_EXTENDIDO.csv")

# Configuración de Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Iniciando GRAN TORNEO SLM en: {DEVICE.upper()}")

# --- 1. PREPARACIÓN DE DATOS ---
def cargar_datos_prueba():
    try:
        # Intentar cargar local o relativo
        if os.path.exists(INPUT_FILE): df = pd.read_csv(INPUT_FILE)
        elif os.path.exists("SILVER_STANDARD_FINAL_RAG.csv"): df = pd.read_csv("SILVER_STANDARD_FINAL_RAG.csv")
        else: return []

        if os.path.exists(ONTOLOGY_FILE):
            with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f: ontologia = json.load(f)
        elif os.path.exists("rafaela_ontology.json"):
            with open("rafaela_ontology.json", 'r', encoding='utf-8') as f: ontologia = json.load(f)
        else: return []

        # Seleccionar 1 caso de cada tipo crítico
        casos = []
        # Normalizar columna texto
        col_txt = next((c for c in ['Observacion','HALLAZGO','TXT'] if c in df.columns), 'Observacion')
        
        for clase in ['SOCIAL', 'BIOTICO', 'DEFICIENCIA_TECNICA']:
            sample = df[df['IA_Clase_Predicha'] == clase].head(1)
            if not sample.empty:
                row = sample.iloc[0]
                casos.append({
                    'CLASE': clase,
                    'HALLAZGO': str(row[col_txt]),
                    'NORMA': str(row.get('IA_Norma_Sugerida', 'Ley 1333')),
                    'ACCION': ontologia.get(clase, {}).get('accion_sugerida', 'Revisar')
                })
        return casos
    except: return []

# --- 2. MOTOR DE GESTIÓN DE MODELOS (PIPELINE DINÁMICO) ---

def limpiar_memoria():
    """Libera RAM y VRAM forzosamente entre modelos."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def ejecutar_modelo(nombre_modelo, id_modelo, casos, tipo="causal"):
    print(f"\n⚡ Cargando [{id_modelo}]: {nombre_modelo}...")
    resultados = []
    start = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
        
        # Selección de arquitectura
        if tipo == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(nombre_modelo, torch_dtype="auto", device_map="auto" if DEVICE=="cuda" else None)
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        else:
            # Causal LM (Llama, Phi, Gemma, Qwen)
            model = AutoModelForCausalLM.from_pretrained(nombre_modelo, torch_dtype="auto", device_map="auto" if DEVICE=="cuda" else None, trust_remote_code=True)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        print(f"   -> Modelo cargado en {time.time()-start:.1f}s. Generando...")

        for c in casos:
            # PROMPT ENGINEERING ADAPTATIVO (FORMATOS DE CHAT)
            system_msg = "Eres un ingeniero experto en fiscalización ambiental. Redacta una observación técnica formal en español."
            user_msg = f"Hallazgo: {c['HALLAZGO']}\nNorma Legal: {c['NORMA']}\nAcción Requerida: {c['ACCION']}\n\nRedacta el párrafo de observación:"
            
            # Adaptación de Templates según familia
            if "Phi" in nombre_modelo:
                prompt = f"<|user|>\n{system_msg}\n{user_msg}<|end|>\n<|assistant|>"
            elif "gemma" in nombre_modelo:
                prompt = f"<start_of_turn>user\n{system_msg}\n{user_msg}<end_of_turn>\n<start_of_turn>model"
            elif "Qwen" in nombre_modelo:
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            elif "TinyLlama" in nombre_modelo:
                prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>"
            else: # T5 o Genérico
                prompt = f"{system_msg} Contexto: {c['HALLAZGO']}. Norma: {c['NORMA']}. Acción: {c['ACCION']}."

            # Generación
            if tipo == "seq2seq":
                out = pipe(prompt, max_length=300, do_sample=False)
                txt = out[0]['generated_text']
            else:
                out = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.3, return_full_text=False)
                txt = out[0]['generated_text']
            
            resultados.append(txt.strip())

        # Destruir modelo
        del pipe, model, tokenizer
        
    except Exception as e:
        print(f"❌ Error en {id_modelo}: {e}")
        resultados = [f"Error: {str(e)[:50]}"] * len(casos)
    
    limpiar_memoria()
    return resultados

# --- 3. EJECUCIÓN MAESTRA ---
def main():
    casos = cargar_datos_prueba()
    if not casos:
        print("❌ No hay datos para probar. Verifique CSV y JSON.")
        return

    print(f"🧪 Iniciando pruebas con {len(casos)} casos críticos.")
    
    # A. BASELINE (Reglas)
    res_reglas = [
        f"OBSERVACIÓN ({c['CLASE']}): Se evidencia {c['HALLAZGO']}. Según {c['NORMA']}, se instruye {c['ACCION']}." 
        for c in casos
    ]

    # B. TORNEO DE MODELOS
    # Lista de Campeones (Nombre en HuggingFace, ID Interno, Tipo)
    # Nota: Si Phi-3 es muy pesado para su RAM, coméntelo. Qwen y Gemma son muy eficientes.
    modelos = [
        ("google/flan-t5-base", "FLAN_T5", "seq2seq"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TINY_LLAMA", "causal"),
        ("google/gemma-2b-it", "GEMMA_2B", "causal"),
        ("Qwen/Qwen2-1.5B-Instruct", "QWEN_1.5B", "causal"),
        ("microsoft/Phi-3-mini-4k-instruct", "PHI_3_MINI", "causal") 
    ]

    resultados_dict = {
        "CLASE": [c['CLASE'] for c in casos],
        "INPUT_HALLAZGO": [c['HALLAZGO'][:50]+"..." for c in casos],
        "A_REGLAS": res_reglas
    }

    for modelo_hf, id_mod, tipo in modelos:
        print(f"--------------------------------------------------")
        res = ejecutar_modelo(modelo_hf, id_mod, casos, tipo)
        resultados_dict[f"MODELO_{id_mod}"] = res

    # C. EXPORTAR RESULTADOS
    df_res = pd.DataFrame(resultados_dict)
    df_res.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n🏆 TORNEO FINALIZADO.")
    print(f"📄 Resultados guardados en: {OUTPUT_FILE}")
    print("   -> Analice qué modelo alucinó menos y respetó mejor la norma.")

if __name__ == "__main__":
    main()