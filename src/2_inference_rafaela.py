# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE INFERENCIA Y GENERACIÓN (RAG Pipeline)
-------------------------------------------------------------------------------------
Descripción:
Aplica el modelo RAFAELA a datos masivos no vistos, generando:
1. Predicciones de Clase y Norma
2. Propuestas de redacción basadas en similitud semántica (RAG) con el Gold Standard.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics.pairwise import cosine_similarity
from safetensors.torch import load_file
import pickle
import tqdm
import sys

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MODEL_PATH = "rafaela_model_v1.safetensors" # El modelo entrenado
TOKENIZER_DIR = "rafaela_tokenizer"
BASE_MODEL_ID = 'dccuchile/bert-base-spanish-wwm-cased'

# Archivos de Datos
FILE_TARGET = "DATASET_MAESTRO_PROTOTIPO_DOCTORAL_V4.csv"
FILE_GOLD = "GOLD_V9.9_261711.csv"
FILE_OUTPUT = "SILVER_STANDARD_RAFAELA.csv"

# --- ARQUITECTURA DEL MODELO (Idéntica al Entrenamiento) ---
class RAFAELA_Network(torch.nn.Module):
    def __init__(self, base_model, n_tax, n_norm):
        super().__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.out_tax = torch.nn.Linear(self.bert.config.hidden_size, n_tax)
        self.out_norm = torch.nn.Linear(self.bert.config.hidden_size, n_norm)
        
    def forward(self, ids, mask):
        out = self.bert(ids, mask)
        cls_token = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        return self.out_tax(x), self.out_norm(x), cls_token

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(str(self.texts[i]), max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}

def construir_prompt(row, col_map):
    # Función auxiliar para crear el texto enriquecido
    parts = []
    for tag, col in col_map.items():
        val = str(row.get(col, '')).strip()
        parts.append(f"[{tag.upper()}] {val}")
    return " ".join(parts)

# --- PIPELINE PRINCIPAL ---
if __name__ == "__main__":
    print("🚀 Iniciando RAFAELA Inference Engine...")
    
    # 1. Carga de Recursos
    print("📂 Cargando artefactos...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    with open("encoder_tax.pkl", "rb") as f: le_tax = pickle.load(f)
    with open("encoder_norm.pkl", "rb") as f: le_norm = pickle.load(f)
    
    # 2. CONFIGURACIÓN QUIRÚRGICA (Surgical Patch)
    # Evita el error de 'size mismatch' ajustando la config manualmente
    print("🔧 Aplicando parche de arquitectura...")
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    config.vocab_size = 250110      # Buffer de seguridad usado en entrenamiento
    config.max_position_embeddings = 514
    config.type_vocab_size = 1
    
    base_model = AutoModel.from_config(config)
    model = RAFAELA_Network(base_model, len(le_tax.classes_), len(le_norm.classes_))
    
    # Cargar pesos
    model.load_state_dict(load_file(MODEL_PATH))
    model.to(DEVICE).eval()
    print("✅ Modelo RAFAELA cargado correctamente.")
    
    # 3. Preparación de Datos
    print("📄 Procesando Datasets...")
    try:
        df_target = pd.read_csv(FILE_TARGET, encoding='utf-8')
        df_gold = pd.read_csv(FILE_GOLD, encoding='utf-8')
    except:
        df_target = pd.read_csv(FILE_TARGET, encoding='latin1')
        df_gold = pd.read_csv(FILE_GOLD, encoding='latin1')

    # Mapeo de Columnas (Ajustar según CSV real)
    map_target = {'sector': 'SECTOR', 'sub': 'Subsector (Automatico)', 'txt': 'Observacion'}
    map_gold = {'sector': 'Sector', 'sub': 'Subsector', 'txt': 'Texto_Mejorado_Humano'}
    
    df_target['input_text'] = df_target.apply(lambda x: construir_prompt(x, map_target), axis=1)
    df_gold['input_text'] = df_gold.apply(lambda x: construir_prompt(x, map_gold), axis=1)

    # 4. Inferencia Vectorial
    def get_vectors(df):
        ds = InferenceDataset(df['input_text'].values, tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        vecs, p_tax, p_norm = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dl):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                pt, pn, emb = model(ids, mask)
                vecs.append(emb.cpu().numpy())
                p_tax.extend(torch.argmax(pt, dim=1).cpu().numpy())
                p_norm.extend(torch.argmax(pn, dim=1).cpu().numpy())
        return np.vstack(vecs), p_tax, p_norm

    print("🧠 Calculando vectores Target...")
    emb_target, pred_tax, pred_norm = get_vectors(df_target)
    print("✨ Calculando vectores Gold Standard...")
    emb_gold, _, _ = get_vectors(df_gold)
    
    # 5. RAG: Búsqueda Semántica
    print("🔍 Ejecutando Búsqueda Semántica (RAG)...")
    sim_matrix = cosine_similarity(emb_target, emb_gold)
    best_matches = np.argmax(sim_matrix, axis=1)
    scores = np.max(sim_matrix, axis=1)
    
    # 6. Consolidación
    df_target['RAFAELA_Clase'] = le_tax.inverse_transform(pred_tax)
    df_target['RAFAELA_Norma'] = le_norm.inverse_transform(pred_norm)
    df_target['RAFAELA_Confianza'] = scores
    
    # Traer la redacción sugerida del Gold Standard
    gold_texts = df_gold.iloc[best_matches]['Texto_Mejorado_Humano'].values
    df_target['RAFAELA_Propuesta_Redaccion'] = gold_texts
    
    df_target.to_csv(FILE_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"✅ ¡Misión Cumplida! Archivo generado: {FILE_OUTPUT}")