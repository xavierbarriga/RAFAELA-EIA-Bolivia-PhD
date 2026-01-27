# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE AUDITORÍA Y VISUALIZACIÓN
-------------------------------------------------------------------------------------
Descripción:
Genera los artefactos gráficos (Figuras) y tablas de evidencia para el documento de Tesis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración Estética Académica
plt.style.use('ggplot')
sns.set_palette("viridis")
OUTPUT_DIR = "RESULTADOS_RAFAELA"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print("📊 Iniciando Auditoría de Resultados RAFAELA...")

# 1. Cargar Datos Generados
df = pd.read_csv("SILVER_STANDARD_RAFAELA.csv")

# 2. Figura 3: Distribución Sectorial Detallada
print("   -> Generando Figura 3 (Comparativa Sectorial)...")
df['SECTOR_CLEAN'] = df['SECTOR'].str.strip().str.upper()
crosstab = pd.crosstab(df['SECTOR_CLEAN'], df['RAFAELA_Clase'])
crosstab_norm = crosstab.div(crosstab.sum(1), axis=0) * 100

ax = crosstab_norm.plot(kind='bar', stacked=True, figsize=(16, 10), colormap='tab20')
plt.title('RAFAELA: Distribución de Tipologías por Sector', fontsize=16)
plt.ylabel('Proporción (%)')
plt.xlabel('Sector Estratégico')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figura3_Sectores_RAFAELA.png", dpi=300)

# 3. Tablas de Evidencia (Anexos)
print("   -> Extrayendo Casos de Alto Impacto (Social/Biótico)...")
cols_show = ['Archivo', 'Observacion', 'RAFAELA_Clase', 'RAFAELA_Propuesta_Redaccion']

# Filtro: Clase Social con alta confianza
evidence_social = df[
    (df['RAFAELA_Clase'] == 'SOCIAL') & (df['RAFAELA_Confianza'] > 0.90)
].head(10)
evidence_social[cols_show].to_csv(f"{OUTPUT_DIR}/Anexo_Evidencia_Social.csv", index=False)

# Filtro: Clase Biótica con alta confianza
evidence_biotic = df[
    (df['RAFAELA_Clase'] == 'BIOTICO') & (df['RAFAELA_Confianza'] > 0.90)
].head(10)
evidence_biotic[cols_show].to_csv(f"{OUTPUT_DIR}/Anexo_Evidencia_Biotica.csv", index=False)

print(f"✅ Auditoría finalizada. Archivos listos en carpeta: {OUTPUT_DIR}")