# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE BENCHMARKING GENERATIVO (Model Comparison)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 1.0

Descripción:
Este script ejecuta un torneo comparativo entre tres enfoques de generación de texto
para redactar observaciones técnicas:
1. Enfoque Determinista (Reglas/Plantillas) - Baseline
2. Enfoque Seq2Seq (Google Flan-T5) - Small Language Model
3. Enfoque Causal (TinyLlama) - Chat Model

El objetivo es demostrar cualitativamente las ventajas del enfoque híbrido frente
a la rigidez de las reglas y las alucinaciones de los LLMs puros.
"""

import pandas as pd
import json
import torch
import gc
import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "SILVER_STANDARD_FINAL_RAG.csv")
ONTOLOGY_FILE = os.path.join(BASE_DIR, "..", "rafaela_ontology.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "RESULTADOS_BENCHMARK_GENERATIVO.csv")

# Configuración de Hardware
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"🚀 Iniciando Benchmark Generativo en: {'GPU' if DEVICE==0 else 'CPU'}")

# --- 1. PREPARACIÓN DE DATOS (MUESTRA REPRESENTATIVA) ---
def cargar_datos_prueba():
    try:
        # Cargar CSV
        if os.path.exists(INPUT_FILE):
            df = pd.read_csv(INPUT_FILE)
        elif os.path.exists("SILVER_STANDARD_FINAL_RAG.csv"):
            df = pd.read_csv("SILVER_STANDARD_FINAL_RAG.csv")
        else:
            raise FileNotFoundError("No se encuentra el dataset Silver Standard.")

        # Cargar Ontología
        if os.path.exists(ONTOLOGY_FILE):
            with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
                ontologia = json.load(f)
        elif os.path.exists("rafaela_ontology.json"):
            with open("rafaela_ontology.json", 'r', encoding='utf-8') as f:
                ontologia = json.load(f)
        else:
            raise FileNotFoundError("No se encuentra la ontología.")

        # Seleccionar casos críticos para la prueba (Uno de cada tipo)
        casos = []
        clases_interes = ['SOCIAL', 'BIOTICO', 'DEFICIENCIA_TECNICA', 'FISICO']
        
        # Normalizar nombre columna texto
        col_txt = next((c for c in ['Observacion','HALLAZGO','TXT'] if c in df.columns), 'Observacion')

        for clase in clases_interes:
            sample = df[df['IA_Clase_Predicha'] == clase].head(1)
            if not sample.empty:
                row = sample.iloc[0]
                casos.append({
                    'CLASE': clase,
                    'HALLAZGO': str(row[col_txt]),
                    'NORMA': str(row.get('IA_Norma_Sugerida', 'Ley 1333')),
                    'ACCION': ontologia.get(clase, {}).get('accion_sugerida', 'Revisar')
                })
        
        print(f"🧪 Casos de prueba seleccionados: {len(casos)}")
        return casos

    except Exception as e:
        print(f"❌ Error preparando datos: {e}")
        sys.exit(1)

# --- 2. DEFINICIÓN DE COMPETIDORES ---

# MODELO A: REGLAS (BASELINE)
def generar_reglas(caso):
    return (f"OBSERVACIÓN TÉCNICA ({caso['CLASE']}): Se identificó {caso['HALLAZGO']}. "
            f"Según {caso['NORMA']}, se debe controlar este riesgo. "
            f"Acción: {caso['ACCION']}")

# MODELO B: FLAN-T5 (SEQ2SEQ)
def correr_flan_t5(casos):
    print("\n🔵 Cargando Modelo B: FLAN-T5-Base (Google)...")
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=DEVICE)
        
        resultados = []
        for c in casos:
            prompt = (
                f"Write a formal technical observation in Spanish. "
                f"Issue: {c['HALLAZGO']}. "
                f"Law: {c['NORMA']}. "
                f"Required Action: {c['ACCION']}."
            )
            out = generator(prompt, max_length=256, do_sample=False)
            resultados.append(out[0]['generated_text'])
        
        # Limpieza
        del model, tokenizer, generator
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return resultados
    except Exception as e:
        return [f"Error T5: {e}"] * len(casos)

# MODELO C: TINYLLAMA (CHAT)
def correr_tinyllama(casos):
    print("\n🟢 Cargando Modelo C: TinyLlama-1.1B-Chat...")
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=DEVICE)
        
        resultados = []
        for c in casos:
            prompt = (
                f"<|system|>\nEres un ingeniero ambiental experto. Redacta una observación formal en español.\n</s>\n"
                f"<|user|>\nRedactar observación:\n"
                f"- Hallazgo: {c['HALLAZGO']}\n"
                f"- Norma: {c['NORMA']}\n"
                f"- Solicitud: {c['ACCION']}\n</s>\n"
                f"<|assistant|>\n"
            )
            out = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            # Limpiar prompt
            full = out[0]['generated_text']
            resp = full.split("<|assistant|>\n")[-1]
            resultados.append(resp)
            
        del model, tokenizer, generator
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return resultados
    except Exception as e:
        return [f"Error Llama: {e}"] * len(casos)

# --- 3. EJECUCIÓN ---
def main():
    casos = cargar_datos_prueba()
    
    # Ejecutar Torneo
    res_reglas = [generar_reglas(c) for c in casos]
    res_t5 = correr_flan_t5(casos)
    res_llama = correr_tinyllama(casos)
    
    # Consolidar Resultados
    print("\n🏆 RESULTADOS DEL BENCHMARK")
    resultados_df = pd.DataFrame({
        'CLASE': [c['CLASE'] for c in casos],
        'INPUT_ORIGINAL': [c['HALLAZGO'][:100]+"..." for c in casos],
        'A_REGLAS': res_reglas,
        'B_FLAN_T5': res_t5,
        'C_TINYLLAMA': res_llama
    })
    
    # Exportar
    resultados_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ Benchmark completado. Archivo generado: {OUTPUT_FILE}")
    print("   -> Utilice este CSV para la Tabla Comparativa en el Capítulo IV.")

if __name__ == "__main__":
    main()