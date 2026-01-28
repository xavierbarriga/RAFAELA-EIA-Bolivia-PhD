# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE AUDITORÍA DE INTEGRIDAD ONTOLÓGICA (QA System)
-------------------------------------------------------------------------------------
Tesis Doctoral: Ciencias Exactas y Tecnología
Autor: Xavier Barriga Sinisterra (PhD Candidate)
Versión: 1.0

Descripción:
Este script ejecuta una batería de pruebas unitarias sobre la Base de Conocimiento
(rafaela_ontology.json) para garantizar la consistencia lógica antes del despliegue.
Detecta:
1. Clases huérfanas (sin descripción).
2. Leyes vacías.
3. Inconsistencias en severidad.
"""

import json
import os
import sys

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_FILE = os.path.join(BASE_DIR, "..", "rafaela_ontology.json")

print("🛡️  Iniciando Protocolo de Validación Ontológica...")

def cargar_ontologia():
    try:
        with open(ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: No se encuentra el archivo de ontología.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("❌ Error: El archivo JSON está corrupto o mal formateado.")
        sys.exit(1)

def validar_estructura(data):
    errores = []
    advertencias = []
    
    campos_obligatorios = ["descripcion", "riesgo_asociado", "palabras_clave", "leyes_vinculantes", "accion_sugerida"]
    
    print(f"   -> Analizando {len(data)} clases taxonómicas...")
    
    for clase, info in data.items():
        # 1. Chequeo de Campos
        for campo in campos_obligatorios:
            if campo not in info:
                errores.append(f"Clase '{clase}' le falta el campo obligatorio: {campo}")
            elif not info[campo]: # Si está vacío
                advertencias.append(f"Clase '{clase}' tiene el campo '{campo}' vacío.")

        # 2. Chequeo de Lógica Legal
        if "leyes_vinculantes" in info:
            leyes = info["leyes_vinculantes"]
            if not isinstance(leyes, list):
                errores.append(f"Clase '{clase}': 'leyes_vinculantes' debe ser una lista.")
            elif len(leyes) == 0 and clase != "RUIDO_DESCARTAR":
                advertencias.append(f"Clase '{clase}' no tiene leyes vinculantes definidas (Riesgo Legal).")

        # 3. Chequeo de Severidad
        if "severidad_default" in info:
            sev = info["severidad_default"]
            if sev not in ["ALTA", "MEDIA", "BAJA", "CRITICA", "NULA"]:
                advertencias.append(f"Clase '{clase}': Severidad '{sev}' no es estándar.")

    return errores, advertencias

def main():
    onto = cargar_ontologia()
    errs, warns = validar_estructura(onto)
    
    print("\n📊 REPORTE DE AUDITORÍA:")
    print("-------------------------")
    
    if errs:
        print(f"🔴 ERRORES CRÍTICOS ({len(errs)}):")
        for e in errs: print(f"   - {e}")
        print("\n❌ LA ONTOLOGÍA NO ES APTA PARA PRODUCCIÓN.")
    else:
        print("✅ No se encontraron errores críticos de estructura.")

    if warns:
        print(f"\n🟡 ADVERTENCIAS ({len(warns)}):")
        for w in warns: print(f"   - {w}")
    else:
        print("✨ La ontología está perfectamente limpia.")

    if not errs:
        print("\n🏆 ESTADO: VALIDADO (Ready for Deployment)")

if __name__ == "__main__":
    main()