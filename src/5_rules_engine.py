# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE VALIDACIÓN LÓGICA Y REGLAS (Rules Engine)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 1.0

Descripción:
Este módulo implementa el componente SIMBÓLICO de la arquitectura Neuro-Simbólica.
Actúa como un "filtro lógico" que recibe las predicciones probabilísticas de la red
neuronal (Script 2) y las valida contra la Ontología (Script 4) y reglas de negocio
deterministas para garantizar la consistencia legal.
"""

import pandas as pd
import json
import os
import sys

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumimos que los datos están en la carpeta padre 'data' o en la raíz
INPUT_FILE = os.path.join(BASE_DIR, "..", "SILVER_STANDARD_FINAL_RAG.csv") 
ONTOLOGY_FILE = os.path.join(BASE_DIR, "..", "rafaela_ontology.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "GOLDEN_STANDARD_NEUROSYMBOLIC.csv")

print("⚖️  Iniciando Motor de Validación Simbólica (Reglas de Negocio)...")

# --- 1. CARGA DE RECURSOS ---
def cargar_datos():
    try:
        if not os.path.exists(INPUT_FILE):
            # Fallback para pruebas locales si no está en la estructura de carpetas
            local_input = "SILVER_STANDARD_FINAL_RAG.csv"
            if os.path.exists(local_input):
                print(f"   -> Cargando datos desde raíz: {local_input}")
                return pd.read_csv(local_input), local_input
            else:
                raise FileNotFoundError(f"No se encuentra el archivo de entrada: {INPUT_FILE}")
        
        print(f"   -> Cargando datos desde: {INPUT_FILE}")
        return pd.read_csv(INPUT_FILE), INPUT_FILE
    except Exception as e:
        print(f"❌ Error cargando CSV: {e}")
        sys.exit(1)

def cargar_ontologia():
    try:
        if not os.path.exists(ONTOLOGY_FILE):
             # Fallback local
            local_onto = "rafaela_ontology.json"
            if os.path.exists(local_onto):
                print(f"   -> Cargando ontología desde raíz: {local_onto}")
                with open(local_onto, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"No se encuentra la ontología: {ONTOLOGY_FILE}")

        print(f"   -> Cargando ontología desde: {ONTOLOGY_FILE}")
        with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error cargando Ontología: {e}")
        sys.exit(1)

# --- 2. LÓGICA DE NEGOCIO (EL NÚCLEO SIMBÓLICO) ---

def aplicar_reglas_expertas(row):
    """
    Función determinista que corrige la normativa sugerida por la IA
    basándose en reglas estrictas del dominio (Sector + Clase).
    """
    # Normalización para evitar errores de mayúsculas/espacios
    sector = str(row.get('SECTOR', '')).strip().upper()
    clase_predicha = str(row.get('IA_Clase_Predicha', '')).strip()
    norma_ia = str(row.get('IA_Norma_Sugerida', ''))
    
    # --- REGLA 1: PRIORIDAD CONSTITUCIONAL EN TEMAS SOCIALES ---
    # Si es un conflicto social en Hidrocarburos, la Consulta Previa es mandatoria.
    if clase_predicha == 'SOCIAL':
        if 'HIDROCARBUROS' in sector:
            return "D.S. 29033 (Reglamento de Consulta y Participación)"
        else:
            # Para otros sectores, aplica la norma marco constitucional
            return "CPE Art. 30 (Derechos de las Naciones y Pueblos Indígena Originario Campesinos)"
            
    # --- REGLA 2: REGLAMENTOS SECTORIALES ESPECÍFICOS (RASH vs RASE) ---
    # Diferencia técnica entre Hidrocarburos y Electricidad para temas operativos
    if clase_predicha in ['FISICO', 'BIOTICO', 'GESTION_OPERATIVA']:
        if 'HIDROCARBUROS' in sector:
            return "R.A.S.H. (Reglamento Ambiental del Sector Hidrocarburos)"
        if 'ELECTRICIDAD' in sector:
            return "R.A.S.E. (Reglamento Ambiental del Sector Eléctrico)"
    
    # --- REGLA 3: PROCEDIMIENTOS ADMINISTRATIVOS ---
    # Las fallas de forma siempre caen en el reglamento general de prevención
    if clase_predicha == 'DEFICIENCIA_TECNICA' or clase_predicha == 'NORMATIVA':
        return "D.S. 3549 (Reglamento de Prevención y Control Ambiental)"
        
    # --- DEFAULT: CONFIANZA EN LA IA ---
    # Si ninguna regla "dura" se activa, aceptamos la sugerencia probabilística de BERT
    return norma_ia

def generar_plantilla_formal(row, ontologia):
    """
    Generador de Texto Basado en Plantillas (Fallback determinista).
    Ensambla la redacción final usando las variables validadas.
    """
    clase = row.get('IA_Clase_Predicha', 'GENERAL')
    norma_validada = row.get('REGLAS_Norma_Validada', 'Normativa Vigente')
    hallazgo = str(row.get('Observacion', '')).strip() # Asumiendo columna 'Observacion' del Silver
    if not hallazgo or hallazgo == 'nan':
        hallazgo = str(row.get('Texto', 'Sin descripción')) # Fallback a columna 'Texto'

    # Recuperar conocimiento de la Ontología
    conocimiento = ontologia.get(clase, {})
    accion = conocimiento.get('accion_sugerida', 'Realizar las correcciones pertinentes.')
    riesgo = conocimiento.get('riesgo_asociado', 'Incumplimiento normativo.')

    # Construcción del texto (Estructura "Abogado/Ingeniero")
    texto = (
        f"OBSERVACIÓN TÉCNICA ({clase}):\n"
        f"1. HALLAZGO: Se ha identificado que {hallazgo}\n"
        f"2. CRITERIO LEGAL: De conformidad con {norma_validada}, el Promotor debe controlar el riesgo de {riesgo}\n"
        f"3. DICTAMEN: Por lo expuesto, se instruye {accion}"
    )
    return texto

# --- 3. EJECUCIÓN DEL PIPELINE ---
def main():
    # A. Carga
    df, path_df = cargar_datos()
    ontologia = cargar_ontologia()
    
    print(f"📊 Procesando {len(df)} registros...")
    
    # B. Aplicación de Reglas (Validación)
    print("⚙️  Aplicando Motor de Inferencia Simbólica...")
    df['REGLAS_Norma_Validada'] = df.apply(aplicar_reglas_expertas, axis=1)
    
    # C. Cálculo de Métricas de Intervención (Auditoría Doctoral)
    # ¿Cuántas veces el sistema de reglas corrigió a la red neuronal?
    cambios = df[df['IA_Norma_Sugerida'] != df['REGLAS_Norma_Validada']].shape[0]
    tasa_intervencion = (cambios / len(df)) * 100
    
    print(f"   -> Intervenciones del Motor de Reglas: {cambios} casos ({tasa_intervencion:.2f}%)")
    print("   -> Esto demuestra la necesidad del enfoque híbrido vs. puramente neuronal.")

    # D. Generación de Texto (Versión Plantilla/Determinista)
    print("✍️  Generando propuestas de redacción formal...")
    df['RAFAELA_Propuesta_Final'] = df.apply(lambda row: generar_plantilla_formal(row, ontologia), axis=1)
    
    # E. Exportación
    try:
        # Guardar en la misma ubicación que el input o en la raíz
        save_path = "GOLDEN_STANDARD_NEUROSYMBOLIC.csv"
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ ¡ÉXITO! Archivo Golden Standard generado: {save_path}")
        print("   Este dataset contiene la fusión de IA + Reglas + Ontología.")
    except Exception as e:
        print(f"❌ Error guardando archivo final: {e}")

if __name__ == "__main__":
    main()