# -*- coding: utf-8 -*-
"""
RAFAELA: Red Artificial de Fiscalización Ambiental Energética y Licenciamiento Asistido
MÓDULO DE GENERACIÓN DE TEXTO CON LLM
-------------------------------------------------------------------------------------
Descripción:
Este módulo extiende RAFAELA con capacidades de generación de texto usando LLMs.
Combina el enfoque RAG existente con generación aumentada por contexto.

Soporta múltiples proveedores:
1. OpenAI (GPT-3.5/GPT-4)
2. Anthropic (Claude)
3. Modelos locales via Ollama (Llama, Mistral, etc.)
4. HuggingFace (FLAN-T5, etc.)

Autor: Xavier Barriga
Proyecto: Tesis Doctoral - EIA Bolivia
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics.pairwise import cosine_similarity
from safetensors.torch import load_file
import pickle
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MODEL_PATH = "rafaela_model_v1.safetensors"
TOKENIZER_DIR = "rafaela_tokenizer"
BASE_MODEL_ID = 'dccuchile/bert-base-spanish-wwm-cased'

# Archivos de Datos
FILE_TARGET = "DATASET_MAESTRO_PROTOTIPO_DOCTORAL_V4.csv"
FILE_GOLD = "GOLD_V9.9_261711.csv"
FILE_OUTPUT_LLM = "SILVER_STANDARD_LLM_GENERATED.csv"


# =============================================================================
# CLASES DE CONFIGURACIÓN
# =============================================================================

@dataclass
class LLMConfig:
    """Configuración para el LLM."""
    provider: str = "openai"  # openai, anthropic, ollama, huggingface
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # Para Ollama u otros endpoints locales
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 0.9


@dataclass
class RAGContext:
    """Contexto recuperado del RAG para generación."""
    observacion_original: str
    sector: str
    subsector: str
    clase_predicha: str
    norma_predicha: str
    gold_text_similar: str
    similitud_score: float
    gold_metadata: Dict


# =============================================================================
# PROMPTS DE SISTEMA Y USUARIO
# =============================================================================

SYSTEM_PROMPT_RAFAELA = """Eres RAFAELA, un asistente experto en Estudios de Impacto Ambiental (EIA) para el sector energético de Bolivia. Tu rol es mejorar la redacción de observaciones técnicas de fiscalización ambiental.

CONTEXTO DEL SISTEMA:
- Trabajas con la normativa ambiental boliviana (Ley 1333, DS 24176, RASH, etc.)
- Debes generar observaciones técnicas precisas, formales y accionables
- Las observaciones deben seguir el formato técnico de la Autoridad Ambiental Competente

TAXONOMÍA DE CLASIFICACIÓN:
- BIOTICO: Impactos a flora, fauna, ecosistemas
- SOCIAL: Impactos a comunidades, participación ciudadana, compensaciones
- FISICO: Impactos a suelo, agua, aire, geomorfología
- DEFICIENCIA_TECNICA: Errores metodológicos, datos incompletos
- GESTION_OPERATIVA: Planes de gestión, monitoreo, seguimiento
- NORMATIVA: Incumplimientos legales específicos
- RUIDO: Observaciones no relevantes o mal clasificadas

INSTRUCCIONES DE REDACCIÓN:
1. Mantén un tono técnico y formal
2. Sé específico y evita ambigüedades
3. Incluye referencias normativas cuando sea pertinente
4. La observación debe ser accionable (el proponente debe saber qué corregir)
5. Usa terminología técnica apropiada para EIA
"""

USER_PROMPT_TEMPLATE = """Mejora la siguiente observación de fiscalización ambiental.

## OBSERVACIÓN ORIGINAL:
{observacion_original}

## CONTEXTO:
- Sector: {sector}
- Subsector: {subsector}
- Clasificación Predicha: {clase_predicha}
- Normativa Aplicable: {norma_predicha}

## EJEMPLO DE REFERENCIA (Gold Standard con {similitud_score:.0%} de similitud):
{gold_text_similar}

## INSTRUCCIONES:
Genera una versión mejorada de la observación que:
1. Sea técnicamente precisa y formal
2. Incluya la normativa aplicable si corresponde ({norma_predicha})
3. Sea clara y accionable para el proponente
4. Mantenga la esencia del hallazgo original
5. Siga el estilo del ejemplo de referencia

## OBSERVACIÓN MEJORADA:
"""


# =============================================================================
# CLASES BASE PARA PROVEEDORES LLM
# =============================================================================

class LLMProvider(ABC):
    """Clase base abstracta para proveedores de LLM."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Genera texto usando el LLM."""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        """Genera texto para múltiples prompts."""
        pass


class OpenAIProvider(LLMProvider):
    """Proveedor para OpenAI GPT models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Instala openai: pip install openai")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        return response.choices[0].message.content

    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        results = []
        for prompt in tqdm(prompts, desc="Generando con OpenAI"):
            try:
                result = self.generate(prompt, system_prompt)
                results.append(result)
                time.sleep(0.1)  # Rate limiting básico
            except Exception as e:
                logger.error(f"Error en generación: {e}")
                results.append("")
        return results


class AnthropicProvider(LLMProvider):
    """Proveedor para Anthropic Claude models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Instala anthropic: pip install anthropic")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        results = []
        for prompt in tqdm(prompts, desc="Generando con Claude"):
            try:
                result = self.generate(prompt, system_prompt)
                results.append(result)
                time.sleep(0.2)  # Rate limiting para Anthropic
            except Exception as e:
                logger.error(f"Error en generación: {e}")
                results.append("")
        return results


class OllamaProvider(LLMProvider):
    """Proveedor para modelos locales via Ollama."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Instala requests: pip install requests")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            }
        )
        return response.json()["response"]

    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        results = []
        for prompt in tqdm(prompts, desc="Generando con Ollama"):
            try:
                result = self.generate(prompt, system_prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error en generación: {e}")
                results.append("")
        return results


class HuggingFaceProvider(LLMProvider):
    """Proveedor para modelos de HuggingFace (local)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            logger.info(f"Cargando modelo HuggingFace: {config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
        except ImportError:
            raise ImportError("Instala transformers: pip install transformers accelerate")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        outputs = self.pipe(
            full_prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True
        )
        generated = outputs[0]["generated_text"]
        # Remover el prompt del output
        return generated[len(full_prompt):].strip()

    def generate_batch(self, prompts: List[str], system_prompt: str = None) -> List[str]:
        results = []
        for prompt in tqdm(prompts, desc="Generando con HuggingFace"):
            try:
                result = self.generate(prompt, system_prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error en generación: {e}")
                results.append("")
        return results


# =============================================================================
# FACTORY PARA CREAR PROVEEDORES
# =============================================================================

def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory para crear el proveedor de LLM según configuración."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "huggingface": HuggingFaceProvider
    }

    if config.provider not in providers:
        raise ValueError(f"Proveedor no soportado: {config.provider}. Usa: {list(providers.keys())}")

    return providers[config.provider](config)


# =============================================================================
# ARQUITECTURA RAFAELA (Reutilizada)
# =============================================================================

class RAFAELA_Network(torch.nn.Module):
    """Red neuronal RAFAELA para clasificación multi-tarea."""

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
    """Dataset para inferencia."""

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(
            str(self.texts[i]),
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten()
        }


# =============================================================================
# PIPELINE PRINCIPAL DE GENERACIÓN
# =============================================================================

class RAFAELAGenerator:
    """
    Pipeline completo de generación de texto con RAFAELA + LLM.

    Combina:
    1. Clasificación multi-tarea (BERT fine-tuned)
    2. RAG (Retrieval-Augmented Generation)
    3. Generación de texto con LLM
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.llm_provider = None
        self.model = None
        self.tokenizer = None
        self.le_tax = None
        self.le_norm = None

    def load_rafaela_model(self):
        """Carga el modelo RAFAELA y sus artefactos."""
        logger.info("Cargando modelo RAFAELA...")

        # Cargar tokenizer y encoders
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        with open("encoder_tax.pkl", "rb") as f:
            self.le_tax = pickle.load(f)
        with open("encoder_norm.pkl", "rb") as f:
            self.le_norm = pickle.load(f)

        # Configurar arquitectura
        config = AutoConfig.from_pretrained(BASE_MODEL_ID)
        config.vocab_size = 250110
        config.max_position_embeddings = 514
        config.type_vocab_size = 1

        base_model = AutoModel.from_config(config)
        self.model = RAFAELA_Network(base_model, len(self.le_tax.classes_), len(self.le_norm.classes_))
        self.model.load_state_dict(load_file(MODEL_PATH))
        self.model.to(DEVICE).eval()

        logger.info("Modelo RAFAELA cargado correctamente.")

    def load_llm(self):
        """Inicializa el proveedor de LLM."""
        logger.info(f"Inicializando LLM: {self.llm_config.provider}/{self.llm_config.model_name}")
        self.llm_provider = create_llm_provider(self.llm_config)
        logger.info("LLM inicializado correctamente.")

    def construir_prompt(self, row: pd.Series, col_map: Dict) -> str:
        """Construye el prompt estructurado para BERT."""
        parts = []
        for tag, col in col_map.items():
            val = str(row.get(col, '')).strip()
            parts.append(f"[{tag.upper()}] {val}")
        return " ".join(parts)

    def get_vectors(self, df: pd.DataFrame) -> Tuple[np.ndarray, List, List]:
        """Obtiene embeddings y predicciones del modelo RAFAELA."""
        ds = InferenceDataset(df['input_text'].values, self.tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        vecs, p_tax, p_norm = [], [], []

        with torch.no_grad():
            for batch in tqdm(dl, desc="Vectorizando"):
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                pt, pn, emb = self.model(ids, mask)
                vecs.append(emb.cpu().numpy())
                p_tax.extend(torch.argmax(pt, dim=1).cpu().numpy())
                p_norm.extend(torch.argmax(pn, dim=1).cpu().numpy())

        return np.vstack(vecs), p_tax, p_norm

    def create_rag_context(
        self,
        row: pd.Series,
        clase: str,
        norma: str,
        gold_row: pd.Series,
        score: float
    ) -> RAGContext:
        """Crea el contexto RAG para generación."""
        return RAGContext(
            observacion_original=str(row.get('Observacion', row.get('Texto_Original_Crudo', ''))),
            sector=str(row.get('SECTOR', row.get('Sector', ''))),
            subsector=str(row.get('Subsector (Automatico)', row.get('Subsector', ''))),
            clase_predicha=clase,
            norma_predicha=norma,
            gold_text_similar=str(gold_row.get('Texto_Mejorado_Humano', '')),
            similitud_score=score,
            gold_metadata={
                'gold_sector': str(gold_row.get('Sector', '')),
                'gold_dimension': str(gold_row.get('G_Bloque_Dimension', '')),
                'gold_norma': str(gold_row.get('N_Norma_Criterio', ''))
            }
        )

    def build_generation_prompt(self, context: RAGContext) -> str:
        """Construye el prompt para generación con LLM."""
        return USER_PROMPT_TEMPLATE.format(
            observacion_original=context.observacion_original,
            sector=context.sector,
            subsector=context.subsector,
            clase_predicha=context.clase_predicha,
            norma_predicha=context.norma_predicha,
            gold_text_similar=context.gold_text_similar,
            similitud_score=context.similitud_score
        )

    def run_pipeline(
        self,
        df_target: pd.DataFrame,
        df_gold: pd.DataFrame,
        use_llm: bool = True,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de clasificación + RAG + generación.

        Args:
            df_target: DataFrame con observaciones a procesar
            df_gold: DataFrame Gold Standard de referencia
            use_llm: Si usar LLM para generación (False = solo RAG)
            sample_size: Procesar solo N muestras (para testing)

        Returns:
            DataFrame con predicciones y texto generado
        """
        # Limitar muestra si se especifica
        if sample_size:
            df_target = df_target.head(sample_size).copy()
            logger.info(f"Procesando muestra de {sample_size} registros")

        # Mapeos de columnas
        map_target = {'sector': 'SECTOR', 'sub': 'Subsector (Automatico)', 'txt': 'Observacion'}
        map_gold = {'sector': 'Sector', 'sub': 'Subsector', 'txt': 'Texto_Mejorado_Humano'}

        # Construir textos de entrada
        logger.info("Construyendo prompts estructurados...")
        df_target['input_text'] = df_target.apply(lambda x: self.construir_prompt(x, map_target), axis=1)
        df_gold['input_text'] = df_gold.apply(lambda x: self.construir_prompt(x, map_gold), axis=1)

        # Obtener vectores
        logger.info("Calculando embeddings Target...")
        emb_target, pred_tax, pred_norm = self.get_vectors(df_target)

        logger.info("Calculando embeddings Gold Standard...")
        emb_gold, _, _ = self.get_vectors(df_gold)

        # RAG: Búsqueda semántica
        logger.info("Ejecutando búsqueda semántica (RAG)...")
        sim_matrix = cosine_similarity(emb_target, emb_gold)
        best_matches = np.argmax(sim_matrix, axis=1)
        scores = np.max(sim_matrix, axis=1)

        # Decodificar predicciones
        df_target['RAFAELA_Clase'] = self.le_tax.inverse_transform(pred_tax)
        df_target['RAFAELA_Norma'] = self.le_norm.inverse_transform(pred_norm)
        df_target['RAFAELA_Confianza'] = scores

        # Texto RAG (recuperado)
        gold_texts = df_gold.iloc[best_matches]['Texto_Mejorado_Humano'].values
        df_target['RAFAELA_RAG_Texto'] = gold_texts

        if use_llm and self.llm_provider:
            # Generación con LLM
            logger.info("Iniciando generación con LLM...")

            # Crear contextos RAG
            contexts = []
            for i, (idx, row) in enumerate(df_target.iterrows()):
                gold_row = df_gold.iloc[best_matches[i]]
                context = self.create_rag_context(
                    row,
                    df_target.loc[idx, 'RAFAELA_Clase'],
                    df_target.loc[idx, 'RAFAELA_Norma'],
                    gold_row,
                    scores[i]
                )
                contexts.append(context)

            # Construir prompts
            prompts = [self.build_generation_prompt(ctx) for ctx in contexts]

            # Generar con LLM
            generated_texts = self.llm_provider.generate_batch(
                prompts,
                system_prompt=SYSTEM_PROMPT_RAFAELA
            )

            df_target['RAFAELA_LLM_Generado'] = generated_texts
            df_target['RAFAELA_LLM_Provider'] = self.llm_config.provider
            df_target['RAFAELA_LLM_Model'] = self.llm_config.model_name
        else:
            logger.info("Modo RAG-only (sin generación LLM)")
            df_target['RAFAELA_LLM_Generado'] = df_target['RAFAELA_RAG_Texto']

        return df_target


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def load_config_from_env() -> LLMConfig:
    """Carga configuración desde variables de entorno."""
    return LLMConfig(
        provider=os.getenv("RAFAELA_LLM_PROVIDER", "openai"),
        model_name=os.getenv("RAFAELA_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("RAFAELA_LLM_API_KEY"),
        temperature=float(os.getenv("RAFAELA_LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("RAFAELA_LLM_MAX_TOKENS", "1024"))
    )


def load_config_from_file(config_path: str) -> LLMConfig:
    """Carga configuración desde archivo JSON."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return LLMConfig(**config_dict)


# =============================================================================
# EJEMPLOS DE USO
# =============================================================================

def ejemplo_openai():
    """Ejemplo usando OpenAI GPT."""
    config = LLMConfig(
        provider="openai",
        model_name="gpt-4o-mini",  # Opciones: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
        temperature=0.3,
        max_tokens=1024
    )
    return config


def ejemplo_anthropic():
    """Ejemplo usando Anthropic Claude."""
    config = LLMConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",  # Opciones: claude-sonnet-4-20250514, claude-3-5-haiku-20241022
        temperature=0.3,
        max_tokens=1024
    )
    return config


def ejemplo_ollama():
    """Ejemplo usando Ollama (modelos locales)."""
    config = LLMConfig(
        provider="ollama",
        model_name="llama3.1:8b",  # Opciones: mistral, mixtral, llama3.1, qwen2.5
        base_url="http://localhost:11434",
        temperature=0.3,
        max_tokens=1024
    )
    return config


def ejemplo_huggingface():
    """Ejemplo usando HuggingFace local."""
    config = LLMConfig(
        provider="huggingface",
        model_name="google/flan-t5-large",  # O cualquier modelo generativo
        temperature=0.3,
        max_tokens=512
    )
    return config


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAFAELA - Generación de Texto con LLM")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic", "ollama", "huggingface"],
                       help="Proveedor de LLM a usar")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Nombre del modelo")
    parser.add_argument("--sample", type=int, default=None,
                       help="Número de muestras a procesar (None = todas)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Solo RAG, sin generación LLM")
    parser.add_argument("--config", type=str, default=None,
                       help="Ruta a archivo de configuración JSON")

    args = parser.parse_args()

    print("=" * 70)
    print("RAFAELA - Sistema de Generación de Texto con LLM")
    print("=" * 70)

    # Configurar LLM
    if args.config:
        llm_config = load_config_from_file(args.config)
    else:
        llm_config = LLMConfig(
            provider=args.provider,
            model_name=args.model
        )

    print(f"\nConfiguración LLM:")
    print(f"  Provider: {llm_config.provider}")
    print(f"  Model: {llm_config.model_name}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Max Tokens: {llm_config.max_tokens}")

    # Inicializar pipeline
    generator = RAFAELAGenerator(llm_config)
    generator.load_rafaela_model()

    if not args.no_llm:
        generator.load_llm()

    # Cargar datos
    print("\nCargando datasets...")
    try:
        df_target = pd.read_csv(FILE_TARGET, encoding='utf-8')
        df_gold = pd.read_csv(FILE_GOLD, encoding='utf-8')
    except:
        df_target = pd.read_csv(FILE_TARGET, encoding='latin1')
        df_gold = pd.read_csv(FILE_GOLD, encoding='latin1')

    print(f"  Target: {len(df_target)} registros")
    print(f"  Gold Standard: {len(df_gold)} registros")

    # Ejecutar pipeline
    print("\nEjecutando pipeline...")
    df_result = generator.run_pipeline(
        df_target,
        df_gold,
        use_llm=not args.no_llm,
        sample_size=args.sample
    )

    # Guardar resultados
    output_file = FILE_OUTPUT_LLM
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResultados guardados en: {output_file}")

    # Mostrar muestra de resultados
    print("\n" + "=" * 70)
    print("MUESTRA DE RESULTADOS:")
    print("=" * 70)

    sample_cols = ['RAFAELA_Clase', 'RAFAELA_Norma', 'RAFAELA_Confianza']
    if 'RAFAELA_LLM_Generado' in df_result.columns:
        sample_cols.append('RAFAELA_LLM_Generado')

    print(df_result[sample_cols].head(3).to_string())

    print("\nPipeline completado exitosamente.")
