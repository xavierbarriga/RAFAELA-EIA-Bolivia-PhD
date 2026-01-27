# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE ENTRENAMIENTO (Training Pipeline)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 1.0 (Release Candidate)

Descripción:
Entrena un modelo BERT Multi-Tarea para:
1. Clasificación Taxonómica (7 Clases)
2. Sugerencia Normativa (Leyes/Decretos)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from safetensors.torch import save_file as save_safetensors
import os
import pickle
import warnings
import random

# --- CONFIGURACIÓN DEL SISTEMA ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARCHIVOS (Rutas Relativas) ---
CSV_INPUT = "GOLD_V9.9_261711.csv" 
MODEL_OUTPUT = "rafaela_model_v1.safetensors"
TOKENIZER_DIR = "rafaela_tokenizer"
ENCODER_TAX = "encoder_tax.pkl"
ENCODER_NORM = "encoder_norm.pkl"

# --- HIPERPARÁMETROS DOCTORALES ---
BATCH_SIZE = 8
MAX_LEN = 256
EPOCHS = 10
LEARNING_RATE = 2e-5

# --- LÓGICA DE NEGOCIO (TAXONOMÍA) ---
def simplificar_taxonomia(bloque):
    bloque = str(bloque).upper().strip()
    if "INGENIERIA" in bloque or "INSTRUMENTO" in bloque: return "DEFICIENCIA_TECNICA"
    if "FISICO" in bloque or "FÍSICO" in bloque: return "FISICO"
    if "GESTION" in bloque or "GESTIÓN" in bloque: return "GESTION_OPERATIVA"
    if "BIOTICO" in bloque or "BIÓTICO" in bloque: return "BIOTICO"
    if "NORMATIVA" in bloque: return "NORMATIVA"
    if "SOCIAL" in bloque: return "SOCIAL"
    return "RUIDO_DESCARTAR"

# --- CLASES DE PROCESAMIENTO ---
class RAFAELA_Processor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, encoding='latin1' if 'latin' in filepath else 'utf-8')
        self.df = self.df.dropna(subset=['Texto_Mejorado_Humano', 'G_Bloque_Dimension'])
        self.le_tax = LabelEncoder()
        self.le_norm = LabelEncoder()
        
    def process(self):
        self.df['Target_Simplificado'] = self.df['G_Bloque_Dimension'].apply(simplificar_taxonomia)
        
        # Ingeniería de Prompts (Contexto Estructurado)
        cols = ['Sector', 'Subsector', 'Descripcion_AOP', 'Tipo_Ingreso', 'Aspectos_Considerar', 
                'Departamento', 'Provincia', 'U_Ubicacion_Ref', 'Texto_Mejorado_Humano']
        for c in cols: self.df[c] = self.df[c].fillna('')
        
        self.df['input_text'] = (
            "[SECTOR] " + self.df['Sector'] + " [SUB] " + self.df['Subsector'] + 
            " [AOP] " + self.df['Descripcion_AOP'] + " [TIPO] " + self.df['Tipo_Ingreso'] + 
            " [LOC] " + self.df['U_Ubicacion_Ref'] + " [TXT] " + self.df['Texto_Mejorado_Humano']
        )
        
        self.df['encoded_tax'] = self.le_tax.fit_transform(self.df['Target_Simplificado'])
        
        # Normalización de Etiquetas Legales (Top 30 + Otros)
        self.df['target_norm'] = self.df['N_Norma_Criterio'].fillna('GENERICO').astype(str)
        top_norm = self.df['target_norm'].value_counts().nlargest(30).index
        self.df['target_norm'] = self.df['target_norm'].apply(lambda x: x if x in top_norm else "OTRA_NORMATIVA")
        self.df['encoded_norm'] = self.le_norm.fit_transform(self.df['target_norm'])
        
        return self.df

class RAFAELA_Dataset(Dataset):
    def __init__(self, texts, tax, norm, tokenizer):
        self.texts = texts
        self.tax = tax
        self.norm = norm
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(
            str(self.texts[i]), max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'tax': torch.tensor(self.tax[i], dtype=torch.long),
            'norm': torch.tensor(self.norm[i], dtype=torch.long)
        }

class RAFAELA_Network(torch.nn.Module):
    def __init__(self, base_model, n_tax, n_norm):
        super().__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.out_tax = torch.nn.Linear(self.bert.config.hidden_size, n_tax)
        self.out_norm = torch.nn.Linear(self.bert.config.hidden_size, n_norm)
        
    def forward(self, ids, mask):
        out = self.bert(ids, mask)
        x = self.dropout(out.last_hidden_state[:, 0, :])
        return self.out_tax(x), self.out_norm(x)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print(f"🚀 Iniciando Entrenamiento de RAFAELA en {DEVICE}")
    
    # 1. Procesamiento
    proc = RAFAELA_Processor(CSV_INPUT)
    df = proc.process()
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['encoded_tax'], random_state=42)
    
    # 2. Pesos para Balanceo de Clases (Solución al sesgo)
    weights = compute_class_weight('balanced', classes=np.unique(df['encoded_tax']), y=df['encoded_tax'])
    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    
    # 3. Configuración del Tokenizer con Tokens Especiales
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    special_tokens = ["[SECTOR]", "[SUB]", "[AOP]", "[TIPO]", "[LOC]", "[TXT]"]
    tokenizer.add_tokens(special_tokens)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # 4. Entrenamiento
    train_ds = RAFAELA_Dataset(df_train.input_text.values, df_train.encoded_tax.values, df_train.encoded_norm.values, tokenizer)
    model = RAFAELA_Network(base_model, len(proc.le_tax.classes_), len(proc.le_norm.classes_)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_tax = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_norm = torch.nn.CrossEntropyLoss()
    
    print("🔥 Entrenando épocas...")
    for epoch in range(EPOCHS):
        model.train()
        dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        for batch in dl:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            tax, norm = model(ids, mask)
            loss = criterion_tax(tax, batch['tax'].to(DEVICE)) + 0.5 * criterion_norm(norm, batch['norm'].to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"   -> Época {epoch+1}/{EPOCHS} completada.")
        
    # 5. Guardado de Artefactos
    print("💾 Guardando modelo RAFAELA...")
    save_safetensors(model.state_dict(), MODEL_OUTPUT)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    with open(ENCODER_TAX, "wb") as f: pickle.dump(proc.le_tax, f)
    with open(ENCODER_NORM, "wb") as f: pickle.dump(proc.le_norm, f)
    print("✅ Proceso finalizado exitosamente.")