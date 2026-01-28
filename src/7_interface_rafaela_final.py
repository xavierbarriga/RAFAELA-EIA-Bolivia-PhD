# ==============================================================================
# PROYECTO RAFAELA (Tesis Doctoral 2026)
# INTERFAZ FINAL v4.0 ("EL PENTÁGONO": Benchmark Multi-Modelo en Tiempo Real)
# ==============================================================================
# Características:
# 1. Carga Dinámica de 5 Arquitecturas SLM (Gestión de VRAM/RAM).
# 2. RAG Contextual con Silver Standard.
# 3. Motor Neuro-Simbólico Completo.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import gc
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importación segura de Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="RAFAELA | Tesis Doctoral", page_icon="🇧🇴", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #003366; color: white; height: 3em; font-weight: bold; width:100%;}
    .var-card {background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .var-val {color: #003366; font-weight: bold; font-size: 1.1em;}
    .rag-card {background-color: #fff; border-left: 4px solid #28a745; padding: 10px; margin-bottom: 8px; font-size: 0.9em; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .model-tag {font-size: 0.8em; font-weight: bold; padding: 2px 6px; border-radius: 4px; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- 1. GESTIÓN DE DATOS ---
@st.cache_data
def cargar_datos_seguro():
    try:
        df = pd.read_csv('SILVER_STANDARD_FINAL_RAG.csv')
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Mapeo y limpieza
        col_texto = next((c for c in ['OBSERVACION', 'HALLAZGO', 'TXT', 'TEXTO_MEJORADO_HUMANO'] if c in df.columns), None)
        df['TEXTO_RAG'] = df[col_texto].fillna("S/D") if col_texto else "S/D"
        
        for c in ['SECTOR', 'SUBSECTOR', 'TIPO_INGRESO', 'DEPARTAMENTO', 'PROVINCIA', 'ASPECTOS_CONSIDERAR']:
            df[c] = df[c].astype(str).str.strip().str.upper() if c in df.columns else "S/D"
            
        return df
    except: return pd.DataFrame()

@st.cache_data
def cargar_ontologia():
    try:
        with open('rafaela_ontology.json', 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

df_db = cargar_datos_seguro()
ontologia = cargar_ontologia()

# --- 2. MOTOR DE GENERACIÓN DINÁMICO (LA MAGIA) ---
def generar_con_ia(modelo_id, prompt_sistema, prompt_usuario):
    """
    Carga un modelo, genera, y lo descarga inmediatamente para ahorrar RAM.
    """
    if not IA_AVAILABLE: return "Librería Transformers no instalada."
    
    resultado = ""
    try:
        # Selección de Arquitectura
        if modelo_id == "FLAN-T5":
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            input_text = f"{prompt_sistema} {prompt_usuario}"
            out = pipe(input_text, max_length=300, do_sample=False)
            resultado = out[0]['generated_text']
            
        else:
            # Modelos Causales (TinyLlama, Qwen, Phi)
            if modelo_id == "TINYLLAMA": name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif modelo_id == "QWEN": name = "Qwen/Qwen2-1.5B-Instruct"
            elif modelo_id == "PHI-3": name = "microsoft/Phi-3-mini-4k-instruct"
            elif modelo_id == "GEMMA": name = "google/gemma-2b-it"
            
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # Formateo de Chat (Simplificado)
            full_prompt = f"System: {prompt_sistema}\nUser: {prompt_usuario}\nAssistant:"
            out = pipe(full_prompt, max_new_tokens=200, do_sample=True, temperature=0.3)
            raw_text = out[0]['generated_text']
            resultado = raw_text.split("Assistant:")[-1].strip()

        # LIMPIEZA DE MEMORIA CRÍTICA
        del model, tokenizer, pipe
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return resultado

    except Exception as e:
        return f"Error cargando {modelo_id}: {str(e)[:50]}... (Posible falta de RAM)"

# --- 3. INTERFAZ: CONTEXTO (CUERPO 1) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Coat_of_arms_of_Bolivia.svg/100px-Coat_of_arms_of_Bolivia.svg.png", width=80)
    st.markdown("### RAFAELA v4.0")
    st.info("Arquitectura Neuro-Simbólica Multi-Modelo")
    rag_ph = st.container()

st.subheader("1. Contexto del Proyecto (Inputs Predictivos)")
with st.container():
    c1, c2, c3 = st.columns(3)
    sector = c1.selectbox("Sector", sorted(df_db['SECTOR'].unique()) if not df_db.empty else [])
    # Filtros cascada
    subs = sorted(df_db[df_db['SECTOR']==sector]['SUBSECTOR'].unique()) if not df_db.empty else []
    subsector = c2.selectbox("Subsector", subs)
    tipo = c3.selectbox("Instrumento", sorted(df_db['TIPO_INGRESO'].unique()) if not df_db.empty else [])
    
    c4, c5, c6 = st.columns(3)
    depto = c4.selectbox("Departamento", sorted(df_db['DEPARTAMENTO'].unique()) if not df_db.empty else [])
    provs = sorted(df_db[df_db['DEPARTAMENTO']==depto]['PROVINCIA'].unique()) if not df_db.empty else []
    provincia = c5.selectbox("Provincia", provs)
    aspecto = c6.selectbox("Factor Ambiental", sorted(df_db['ASPECTOS_CONSIDERAR'].unique())[:50] if not df_db.empty else [])

# --- 4. INTERFAZ: LABORATORIO (CUERPO 2) ---
st.markdown("---")
st.subheader("2. Laboratorio de Generación (F + U -> Multi-Model)")

col_in, col_out = st.columns([1, 2])

with col_in:
    st.info("Variables de Entrada:")
    var_u = st.text_input("📍 Variable U (Ubicación)", placeholder="Ej: Pozo SAL-14")
    var_f = st.text_area("📝 Variable F (Facto)", height=150, placeholder="Ej: Se observa falta de señalización...")
    
    # Selector de Modelos para la Demo
    st.markdown("**Seleccione Modelos a Ejecutar:**")
    run_rules = st.checkbox("Reglas (Baseline)", value=True)
    run_t5 = st.checkbox("Google Flan-T5 (Seq2Seq)", value=True)
    run_llama = st.checkbox("TinyLlama 1.1B (Causal)", value=False)
    run_qwen = st.checkbox("Qwen 1.5B (SOTA)", value=False)
    
    btn_calc = st.button("🚀 EJECUTAR RAFAELA")

# LÓGICA DE PROCESAMIENTO
if btn_calc and var_f:
    
    # A. INFERENCIA DE VARIABLES (SIMULADA PARA VELOCIDAD UI)
    txt_low = var_f.lower()
    if "social" in txt_low: bloq = "SOCIAL"
    elif "fauna" in txt_low or "flora" in txt_low: bloq = "BIOTICO"
    elif "agua" in txt_low: bloq = "FISICO"
    else: bloq = "DEFICIENCIA_TECNICA"
    
    data_onto = ontologia.get(bloq, {})
    norma_n = data_onto.get("leyes_vinculantes", ["Ley 1333"])[0]
    riesgo_s = data_onto.get("riesgo_asociado", "Riesgo General")
    accion_cat = data_onto.get("accion_sugerida", "Subsanar")
    
    if sector == "HIDROCARBUROS" and bloq == "SOCIAL": norma_n = "D.S. 29033 (Consulta)"

    with col_out:
        # Mostrar Variables Latentes
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clase (BLOQ)", bloq)
        k2.metric("Norma (N)", norma_n.split('(')[0][:10])
        k3.metric("Riesgo (S)", "ALTA")
        k4.metric("Acción (CAT)", "CORREGIR")
        
        st.markdown("### 🤖 Propuestas de Redacción (Benchmark)")
        
        # B. GENERACIÓN MULTI-MODELO
        tabs = st.tabs(["🔒 Reglas", "🔵 Flan-T5", "🦙 TinyLlama", "🐉 Qwen"])
        
        # 1. Reglas
        with tabs[0]:
            if run_rules:
                txt = f"OBSERVACIÓN ({bloq}): Se evidenció en {var_u}: {var_f}. Incumple {norma_n}. Riesgo: {riesgo_s}. Instrucción: {accion_cat}."
                st.info(txt)
                st.caption("✅ Determinista. 100% Seguro.")
            else: st.write("No seleccionado.")

        # 2. Flan-T5
        with tabs[1]:
            if run_t5:
                with st.spinner("Ejecutando Google T5..."):
                    sys_p = "Redactar observacion tecnica formal ambiental."
                    usr_p = f"Hecho: {var_f}. Norma: {norma_n}. Accion: {accion_cat}."
                    res = generar_con_ia("FLAN-T5", sys_p, usr_p)
                    st.success(res)
                    st.caption("⚡ Rápido y conciso.")
            else: st.write("No seleccionado.")

        # 3. TinyLlama
        with tabs[2]:
            if run_llama:
                with st.spinner("Cargando TinyLlama (Puede tardar)..."):
                    sys_p = "Eres un ingeniero experto. Redacta una observación formal en español."
                    usr_p = f"Hallazgo: {var_f}. Base Legal: {norma_n}. Solicitud: {accion_cat}."
                    res = generar_con_ia("TINYLLAMA", sys_p, usr_p)
                    st.warning(res)
                    st.caption("🎨 Creativo. Verificar alucinaciones.")
            else: st.write("No seleccionado.")

        # 4. Qwen
        with tabs[3]:
            if run_qwen:
                with st.spinner("Cargando Qwen (SOTA)..."):
                    sys_p = "Actúa como fiscalizador ambiental. Redacta observación técnica."
                    usr_p = f"Problema: {var_f} en {var_u}. Ley: {norma_n}. Requerimiento: {accion_cat}."
                    res = generar_con_ia("QWEN", sys_p, usr_p)
                    st.success(res)
                    st.caption("🏆 Mejor en Español.")
            else: st.write("No seleccionado.")

    # --- RAG SIDEBAR ---
    with rag_ph:
        if not df_db.empty:
            subset = df_db[df_db['SECTOR'] == sector].head(2000)
            docs = subset['TEXTO_RAG'].tolist()
            docs.insert(0, var_f)
            tfidf = TfidfVectorizer(stop_words='english').fit_transform(docs)
            sim = cosine_similarity(tfidf[0:1], tfidf).flatten()
            
            st.markdown(f"**Precedentes en {sector}:**")
            for i in sim.argsort()[:-4:-1]:
                if i == 0 or sim[i] < 0.1: continue
                row = subset.iloc[i-1]
                st.markdown(f"<div class='rag-card'><b>{sim[i]:.0%}</b>: {row['TEXTO_RAG'][:100]}...<br><i>Norma: {row.get('N_NORMA_CRITERIO', 'S/D')[:20]}</i></div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<center>RAFAELA v4.0 - Tesis Doctoral 2026</center>", unsafe_allow_html=True)