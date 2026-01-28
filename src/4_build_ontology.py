# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE INGENIERÍA DEL CONOCIMIENTO (Ontology Builder)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 1.0

Descripción:
Este script formaliza el conocimiento experto del dominio ambiental en una estructura
JSON (Ontología Ligera). Define las reglas, riesgos y acciones obligatorias para cada
clase taxonómica, actuando como la base de conocimiento para el Motor Simbólico.
"""

import json
import os
import sys

# Configuración de Rutas (Relativas para GitHub)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "rafaela_ontology.json") # Se guarda en la raíz del repo

print("🏗️  Iniciando Ingeniería del Conocimiento (Ontología RAFAELA)...")

# --- DEFINICIÓN DE LA ONTOLOGÍA (BASE DE CONOCIMIENTO EXPERTA) ---
# Esta estructura representa la cristalización del diagnóstico del sector (Capítulo I Tesis).
# Mapea: CLASE -> {Definición, Riesgo, Keywords, Leyes, Acción}

ONTOLOGIA_RAFAELA = {
    "BIOTICO": {
        "descripcion": "Impactos potenciales sobre flora, fauna, ecosistemas y biodiversidad.",
        "riesgo_asociado": "Daño irreversible al patrimonio natural y biodiversidad.",
        "palabras_clave": [
            "fauna", "flora", "vegetación", "desmonte", "avifauna", "nidos",
            "bosque", "especies", "hábitat", "tala", "animales", "biota", "bofedales"
        ],
        "leyes_vinculantes": [
            "Ley 1333 (Medio Ambiente) - Art. 20",
            "Ley 1700 (Forestal)",
            "Libro Rojo de Vertebrados de Bolivia",
            "D.S. 24176 (Reglamento de Gestión Ambiental)"
        ],
        "severidad_default": "ALTA",
        "accion_sugerida": "Solicitar medidas de mitigación biológica específicas y cronograma de rescate."
    },

    "SOCIAL": {
        "descripcion": "Relacionamiento comunitario, consulta pública y derechos de pueblos indígenas.",
        "riesgo_asociado": "Conflictividad social, bloqueo de operaciones y vulneración de derechos.",
        "palabras_clave": [
            "comunidad", "consulta", "acta", "reunión", "indígena", "tco",
            "compensación", "afectación social", "pueblos", "campesinos", "otb", "capitanía"
        ],
        "leyes_vinculantes": [
            "CPE Art. 30 (Derechos de las Naciones y Pueblos Indígena Originario Campesinos)",
            "D.S. 29033 (Reglamento de Consulta y Participación - Hidrocarburos)",
            "Ley 031 (Marco de Autonomías)",
            "Convenio 169 OIT"
        ],
        "severidad_default": "CRITICA",
        "accion_sugerida": "Verificar actas de validación social y cumplimiento estricto del proceso de Consulta Previa."
    },

    "FISICO": {
        "descripcion": "Impactos sobre factores abióticos: suelo, agua, aire y ruido.",
        "riesgo_asociado": "Contaminación de recursos vitales, erosión y pasivos ambientales.",
        "palabras_clave": [
            "suelo", "agua", "río", "aire", "erosión", "sedimentos", "pozo",
            "residuos", "emisiones", "ruido", "polvo", "cauce", "acuífero", "cuerpos de agua"
        ],
        "leyes_vinculantes": [
            "Ley 1333 - Reglamento en Materia de Contaminación Hídrica (RMCH)",
            "Ley 1333 - Reglamento en Materia de Contaminación Atmosférica (RMCA)",
            "Ley 755 (Gestión Integral de Residuos)",
            "Reglamento Ambiental del Sector (RASH/RASE)"
        ],
        "severidad_default": "MEDIA",
        "accion_sugerida": "Solicitar parámetros de monitoreo físico-químico acreditados y medidas de control de erosión."
    },

    "DEFICIENCIA_TECNICA": {
        "descripcion": "Errores de ingeniería, inconsistencias en datos, mapas, coordenadas o cronogramas.",
        "riesgo_asociado": "Inviabilidad técnica del proyecto o falta de información suficiente para evaluar.",
        "palabras_clave": [
            "cronograma", "coordenadas", "plano", "diseño", "utm", "mapa",
            "ubicación", "descripción", "potencia", "voltaje", "profundidad",
            "diámetro", "trazo", "vértice", "polígono"
        ],
        "leyes_vinculantes": [
            "D.S. 3549 (Reglamento de Prevención y Control Ambiental)",
            "Guías Técnicas del Ministerio de Hidrocarburos y Energías"
        ],
        "severidad_default": "MEDIA",
        "accion_sugerida": "Solicitar aclaración, complementación o enmienda técnica del documento con respaldo ingenieril."
    },

    "NORMATIVA": {
        "descripcion": "Incumplimiento de formalidades administrativas, firmas, vigencia de licencias y requisitos legales.",
        "riesgo_asociado": "Nulidad administrativa del trámite y observación legal.",
        "palabras_clave": [
            "firma", "notario", "declaración jurada", "vigencia", "renca",
            "consultor", "registro", "formato", "digital", "cd", "fojas", "legal", "representante"
        ],
        "leyes_vinculantes": [
            "D.S. 3549 (Procedimientos Administrativos Ambientales)",
            "Ley 2341 (Procedimiento Administrativo)",
            "Resoluciones Administrativas Vigentes"
        ],
        "severidad_default": "BAJA",
        "accion_sugerida": "Exigir subsanación inmediata de requisitos formales y legales."
    },

    "GESTION_OPERATIVA": {
        "descripcion": "Aspectos logísticos del plan de manejo ambiental, seguridad industrial y contingencias.",
        "riesgo_asociado": "Fallas operativas en la implementación del PPM-PASA y seguridad.",
        "palabras_clave": [
            "monitoreo", "informe", "plan de contingencia", "seguridad", "epp",
            "señalización", "residuos sólidos", "campamento", "transporte", "logística"
        ],
        "leyes_vinculantes": [
            "Ley 16998 (Ley General de Higiene y Seguridad Ocupacional)",
            "Reglamento de Prevención y Control Ambiental"
        ],
        "severidad_default": "MEDIA",
        "accion_sugerida": "Ajustar el cronograma y presupuesto del Plan de Aplicación y Seguimiento Ambiental."
    },
    
    "RUIDO_DESCARTAR": {
        "descripcion": "Texto irrelevante, saludos protocolares o fragmentos sin valor técnico.",
        "riesgo_asociado": "Ninguno (Información espuria).",
        "palabras_clave": [],
        "leyes_vinculantes": [],
        "severidad_default": "NULA",
        "accion_sugerida": "Descartar o archivar."
    }
}

# --- GUARDADO DEL ARCHIVO MAESTRO ---
def guardar_ontologia():
    try:
        # Guardar en formato JSON bonito (indentado) para lectura humana
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(ONTOLOGIA_RAFAELA, f, indent=4, ensure_ascii=False)
        
        print(f"✅ ¡ÉXITO! Ontología generada correctamente.")
        print(f"📄 Archivo guardado en: {os.path.abspath(OUTPUT_FILE)}")
        print(f"📊 Clases Definidas: {len(ONTOLOGIA_RAFAELA.keys())}")
        print("💡 Este archivo es el input crítico para el Motor Simbólico (Script 5).")
        
    except Exception as e:
        print(f"❌ Error fatal guardando la ontología: {e}")

# --- VERIFICACIÓN DE INTEGRIDAD ---
def verificar_lectura():
    print("\n🔍 Verificando integridad de datos (Self-Test)...")
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Test rápido: Verificar clase SOCIAL
        desc = data.get('SOCIAL', {}).get('descripcion', 'ERROR')
        leyes = data.get('SOCIAL', {}).get('leyes_vinculantes', [])
        
        print(f"   -> Test Clase SOCIAL: OK")
        print(f"   -> Descripción: {desc[:50]}...")
        print(f"   -> Leyes cargadas: {len(leyes)}")
        
    except Exception as e:
        print(f"❌ Fallo en verificación de lectura: {e}")

if __name__ == "__main__":
    guardar_ontologia()
    verificar_lectura()