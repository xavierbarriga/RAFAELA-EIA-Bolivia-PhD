"""
================================================================================
RAFAELA - Módulo de Generación de Texto con Temas
================================================================================
Sistema de generación de texto estructurado para Estudios de Impacto Ambiental
con soporte para múltiples temas y estilos de redacción técnica.

Autor: Xavier Barriga (PhD Candidate)
Versión: 1.0.0
Framework: Arquitectura Neuro-Simbólica RAFAELA
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime


# ==============================================================================
# ENUMERACIONES Y CONSTANTES
# ==============================================================================

class TipoRedaccion(Enum):
    """Tipos de redacción disponibles en RAFAELA."""
    TECNICO = "tecnico"           # Redacción técnica formal
    NORMATIVO = "normativo"       # Énfasis en marco legal
    EJECUTIVO = "ejecutivo"       # Resumen ejecutivo
    DETALLADO = "detallado"       # Análisis detallado completo
    CIUDADANO = "ciudadano"       # Lenguaje accesible


class NivelFormalidad(Enum):
    """Niveles de formalidad para la redacción."""
    ALTO = "alto"                 # Documentos oficiales
    MEDIO = "medio"               # Informes técnicos
    BAJO = "bajo"                 # Comunicaciones internas


class CategoriaImpacto(Enum):
    """Categorías de impacto ambiental según clasificación RAFAELA."""
    FISICO = "FISICO"
    BIOTICO = "BIOTICO"
    SOCIAL = "SOCIAL"
    NORMATIVA = "NORMATIVA"
    GESTION_OPERATIVA = "GESTION_OPERATIVA"
    DEFICIENCIA_TECNICA = "DEFICIENCIA_TECNICA"


# ==============================================================================
# CONFIGURACIÓN DE TEMA BASE
# ==============================================================================

@dataclass
class ThemeConfig:
    """
    Configuración base para temas de generación de texto.

    Permite personalizar el estilo, formato y estructura
    de las redacciones generadas por RAFAELA.
    """

    # Identificador del tema
    nombre: str = "default"
    version: str = "1.0"

    # Configuración de estilo
    tipo_redaccion: TipoRedaccion = TipoRedaccion.TECNICO
    nivel_formalidad: NivelFormalidad = NivelFormalidad.ALTO

    # Configuración de estructura
    incluir_encabezado: bool = True
    incluir_fundamentacion_legal: bool = True
    incluir_recomendaciones: bool = True
    incluir_plazos: bool = True
    incluir_referencias_cruzadas: bool = False

    # Configuración de formato
    max_caracteres: int = 2000
    usar_viñetas: bool = True
    usar_numeracion: bool = True
    formato_fecha: str = "%d de %B de %Y"
    idioma: str = "es_BO"

    # Tokens especiales para prompt engineering
    tokens_contexto: List[str] = field(default_factory=lambda: [
        "[SECTOR]", "[SUB]", "[AOP]", "[TIPO]", "[LOC]", "[TXT]"
    ])

    # Configuración normativa boliviana
    normativa_principal: str = "Ley 1333 de Medio Ambiente"
    decreto_reglamentario: str = "D.S. 3549"
    autoridad_competente: str = "Autoridad Ambiental Competente Nacional (AACN)"

    def to_dict(self) -> Dict:
        """Convierte la configuración a diccionario."""
        return {
            "nombre": self.nombre,
            "version": self.version,
            "tipo_redaccion": self.tipo_redaccion.value,
            "nivel_formalidad": self.nivel_formalidad.value,
            "incluir_encabezado": self.incluir_encabezado,
            "incluir_fundamentacion_legal": self.incluir_fundamentacion_legal,
            "incluir_recomendaciones": self.incluir_recomendaciones,
            "max_caracteres": self.max_caracteres,
            "idioma": self.idioma
        }


# ==============================================================================
# TEMAS PREDEFINIDOS
# ==============================================================================

class TemasRAFAELA:
    """Colección de temas predefinidos para diferentes contextos de EIA."""

    @staticmethod
    def tema_oficial() -> ThemeConfig:
        """
        Tema para documentos oficiales de la Autoridad Ambiental.
        Máxima formalidad y cumplimiento normativo estricto.
        """
        return ThemeConfig(
            nombre="oficial_aacn",
            version="1.0",
            tipo_redaccion=TipoRedaccion.NORMATIVO,
            nivel_formalidad=NivelFormalidad.ALTO,
            incluir_encabezado=True,
            incluir_fundamentacion_legal=True,
            incluir_recomendaciones=True,
            incluir_plazos=True,
            incluir_referencias_cruzadas=True,
            max_caracteres=3000,
            usar_viñetas=True,
            usar_numeracion=True
        )

    @staticmethod
    def tema_tecnico() -> ThemeConfig:
        """
        Tema para informes técnicos especializados.
        Énfasis en precisión técnica y detalles específicos.
        """
        return ThemeConfig(
            nombre="tecnico_especializado",
            version="1.0",
            tipo_redaccion=TipoRedaccion.TECNICO,
            nivel_formalidad=NivelFormalidad.MEDIO,
            incluir_encabezado=True,
            incluir_fundamentacion_legal=True,
            incluir_recomendaciones=True,
            incluir_plazos=False,
            max_caracteres=2500,
            usar_viñetas=True,
            usar_numeracion=True
        )

    @staticmethod
    def tema_ejecutivo() -> ThemeConfig:
        """
        Tema para resúmenes ejecutivos.
        Conciso y orientado a la toma de decisiones.
        """
        return ThemeConfig(
            nombre="resumen_ejecutivo",
            version="1.0",
            tipo_redaccion=TipoRedaccion.EJECUTIVO,
            nivel_formalidad=NivelFormalidad.MEDIO,
            incluir_encabezado=True,
            incluir_fundamentacion_legal=False,
            incluir_recomendaciones=True,
            incluir_plazos=True,
            max_caracteres=800,
            usar_viñetas=True,
            usar_numeracion=False
        )

    @staticmethod
    def tema_consulta_publica() -> ThemeConfig:
        """
        Tema para documentos de consulta pública.
        Lenguaje accesible para comunidades y ciudadanos.
        """
        return ThemeConfig(
            nombre="consulta_publica",
            version="1.0",
            tipo_redaccion=TipoRedaccion.CIUDADANO,
            nivel_formalidad=NivelFormalidad.BAJO,
            incluir_encabezado=True,
            incluir_fundamentacion_legal=False,
            incluir_recomendaciones=True,
            incluir_plazos=True,
            max_caracteres=1500,
            usar_viñetas=True,
            usar_numeracion=False,
            autoridad_competente="Ministerio de Medio Ambiente y Agua"
        )

    @staticmethod
    def tema_detallado() -> ThemeConfig:
        """
        Tema para análisis detallados completos.
        Máximo nivel de detalle y fundamentación.
        """
        return ThemeConfig(
            nombre="analisis_detallado",
            version="1.0",
            tipo_redaccion=TipoRedaccion.DETALLADO,
            nivel_formalidad=NivelFormalidad.ALTO,
            incluir_encabezado=True,
            incluir_fundamentacion_legal=True,
            incluir_recomendaciones=True,
            incluir_plazos=True,
            incluir_referencias_cruzadas=True,
            max_caracteres=5000,
            usar_viñetas=True,
            usar_numeracion=True
        )


# ==============================================================================
# TEMPLATES DE REDACCIÓN
# ==============================================================================

class TemplateBase(ABC):
    """Clase base abstracta para templates de redacción."""

    def __init__(self, config: ThemeConfig):
        self.config = config

    @abstractmethod
    def generar_encabezado(self, datos: Dict) -> str:
        """Genera el encabezado de la observación."""
        pass

    @abstractmethod
    def generar_cuerpo(self, datos: Dict) -> str:
        """Genera el cuerpo principal de la observación."""
        pass

    @abstractmethod
    def generar_fundamentacion(self, datos: Dict) -> str:
        """Genera la fundamentación legal."""
        pass

    @abstractmethod
    def generar_recomendacion(self, datos: Dict) -> str:
        """Genera las recomendaciones."""
        pass

    def generar_completo(self, datos: Dict) -> str:
        """Genera la redacción completa según la configuración del tema."""
        partes = []

        if self.config.incluir_encabezado:
            partes.append(self.generar_encabezado(datos))

        partes.append(self.generar_cuerpo(datos))

        if self.config.incluir_fundamentacion_legal:
            partes.append(self.generar_fundamentacion(datos))

        if self.config.incluir_recomendaciones:
            partes.append(self.generar_recomendacion(datos))

        texto_completo = "\n\n".join(filter(None, partes))

        # Aplicar límite de caracteres
        if len(texto_completo) > self.config.max_caracteres:
            texto_completo = texto_completo[:self.config.max_caracteres-3] + "..."

        return texto_completo


class TemplateTecnico(TemplateBase):
    """Template para redacción técnica formal."""

    def generar_encabezado(self, datos: Dict) -> str:
        sector = datos.get('sector', 'No especificado')
        subsector = datos.get('subsector', 'No especificado')
        aop = datos.get('aop', 'No especificado')

        return f"""OBSERVACIÓN TÉCNICA - {sector.upper()}
Subsector: {subsector}
Actividad/Obra/Proyecto: {aop}
Clasificación: {datos.get('clasificacion', 'Por determinar')}"""

    def generar_cuerpo(self, datos: Dict) -> str:
        texto_base = datos.get('texto_observacion', '')
        ubicacion = datos.get('ubicacion', '')

        cuerpo = f"Se ha identificado la siguiente observación técnica:\n\n{texto_base}"

        if ubicacion:
            cuerpo += f"\n\nUbicación de referencia: {ubicacion}"

        return cuerpo

    def generar_fundamentacion(self, datos: Dict) -> str:
        norma = datos.get('norma_sugerida', self.config.normativa_principal)

        return f"""FUNDAMENTACIÓN LEGAL:
De conformidad con lo establecido en {norma} y su reglamentación
contenida en el {self.config.decreto_reglamentario}, la presente
observación se fundamenta en el incumplimiento de los requisitos
técnicos y normativos aplicables al sector."""

    def generar_recomendacion(self, datos: Dict) -> str:
        tipo_accion = datos.get('tipo_accion', 'complementar información')

        recomendacion = f"""RECOMENDACIÓN:
Se solicita al Representante Legal {tipo_accion}, de acuerdo
a los lineamientos establecidos por la {self.config.autoridad_competente}."""

        if self.config.incluir_plazos:
            recomendacion += "\n\nPlazo: Según lo establecido en el procedimiento administrativo vigente."

        return recomendacion


class TemplateNormativo(TemplateBase):
    """Template con énfasis en el marco normativo."""

    def generar_encabezado(self, datos: Dict) -> str:
        return f"""OBSERVACIÓN NORMATIVA
Referencia: {datos.get('referencia', 'OBS-' + datetime.now().strftime('%Y%m%d'))}
Sector: {datos.get('sector', 'No especificado')}
Categoría de Impacto: {datos.get('clasificacion', 'Por determinar')}"""

    def generar_cuerpo(self, datos: Dict) -> str:
        return f"""DESCRIPCIÓN DE LA OBSERVACIÓN:

{datos.get('texto_observacion', 'Sin descripción.')}

Aspectos a considerar:
{datos.get('aspectos', '- Verificar cumplimiento normativo')}"""

    def generar_fundamentacion(self, datos: Dict) -> str:
        norma = datos.get('norma_sugerida', self.config.normativa_principal)
        articulos = datos.get('articulos_aplicables', 'artículos pertinentes')

        return f"""MARCO NORMATIVO APLICABLE:

1. {self.config.normativa_principal}
   - {articulos}

2. {self.config.decreto_reglamentario}
   - Procedimientos de Evaluación de Impacto Ambiental

3. Normativa específica: {norma}

La observación se sustenta en el análisis sistemático del marco
regulatorio ambiental vigente en el Estado Plurinacional de Bolivia."""

    def generar_recomendacion(self, datos: Dict) -> str:
        return f"""ACCIONES REQUERIDAS:

a) Presentar documentación complementaria que demuestre el
   cumplimiento de la normativa citada.

b) Implementar las medidas correctivas necesarias según lo
   establecido en el Plan de Manejo Ambiental.

c) Coordinar con la {self.config.autoridad_competente} para
   la verificación de las acciones implementadas."""


class TemplateEjecutivo(TemplateBase):
    """Template para resúmenes ejecutivos concisos."""

    def generar_encabezado(self, datos: Dict) -> str:
        return f"[{datos.get('clasificacion', 'OBSERVACIÓN')}] {datos.get('sector', '')}"

    def generar_cuerpo(self, datos: Dict) -> str:
        texto = datos.get('texto_observacion', '')
        # Resumir a primera oración significativa
        if len(texto) > 200:
            texto = texto[:200].rsplit(' ', 1)[0] + "..."
        return texto

    def generar_fundamentacion(self, datos: Dict) -> str:
        return f"Base legal: {datos.get('norma_sugerida', self.config.normativa_principal)}"

    def generar_recomendacion(self, datos: Dict) -> str:
        return f"Acción: {datos.get('tipo_accion', 'Requiere atención')}"


class TemplateCiudadano(TemplateBase):
    """Template en lenguaje accesible para consulta pública."""

    def generar_encabezado(self, datos: Dict) -> str:
        clasificacion = datos.get('clasificacion', 'Observación')
        mapeo_ciudadano = {
            'FISICO': 'Impacto en el Ambiente Físico',
            'BIOTICO': 'Impacto en Flora y Fauna',
            'SOCIAL': 'Impacto en la Comunidad',
            'NORMATIVA': 'Cumplimiento de Normas',
            'GESTION_OPERATIVA': 'Gestión del Proyecto',
            'DEFICIENCIA_TECNICA': 'Aspectos Técnicos'
        }
        return f"Tipo de Observación: {mapeo_ciudadano.get(clasificacion, clasificacion)}"

    def generar_cuerpo(self, datos: Dict) -> str:
        texto = datos.get('texto_observacion', '')
        # Simplificar lenguaje técnico
        texto = self._simplificar_texto(texto)

        return f"""¿Qué se observó?
{texto}

¿Dónde?
{datos.get('ubicacion', 'Zona del proyecto')}"""

    def generar_fundamentacion(self, datos: Dict) -> str:
        return ""  # Omitido en versión ciudadana

    def generar_recomendacion(self, datos: Dict) -> str:
        return f"""¿Qué se debe hacer?
La empresa responsable debe {datos.get('tipo_accion', 'atender esta observación')}
para proteger el medio ambiente y la comunidad."""

    def _simplificar_texto(self, texto: str) -> str:
        """Simplifica el texto técnico para hacerlo más accesible."""
        reemplazos = {
            'Representante Legal': 'la empresa',
            'EsIA': 'estudio ambiental',
            'EEIA': 'estudio ambiental',
            'PMA': 'plan de manejo',
            'AAC': 'autoridad ambiental',
            'D.S.': 'decreto',
            'incumplimiento normativo': 'no se cumplió con las normas'
        }
        for tecnico, simple in reemplazos.items():
            texto = texto.replace(tecnico, simple)
        return texto


# ==============================================================================
# GENERADOR DE TEXTO PRINCIPAL
# ==============================================================================

class TextGenerator:
    """
    Generador principal de texto para RAFAELA.

    Integra la configuración de temas con los templates
    para producir redacciones estructuradas.
    """

    def __init__(self, config: Optional[ThemeConfig] = None):
        """
        Inicializa el generador con una configuración de tema.

        Args:
            config: Configuración del tema. Si es None, usa el tema por defecto.
        """
        self.config = config or ThemeConfig()
        self.template = self._seleccionar_template()

    def _seleccionar_template(self) -> TemplateBase:
        """Selecciona el template apropiado según el tipo de redacción."""
        mapeo_templates = {
            TipoRedaccion.TECNICO: TemplateTecnico,
            TipoRedaccion.NORMATIVO: TemplateNormativo,
            TipoRedaccion.EJECUTIVO: TemplateEjecutivo,
            TipoRedaccion.CIUDADANO: TemplateCiudadano,
            TipoRedaccion.DETALLADO: TemplateTecnico,  # Usa técnico con más caracteres
        }
        template_class = mapeo_templates.get(
            self.config.tipo_redaccion,
            TemplateTecnico
        )
        return template_class(self.config)

    def generar(self, datos: Dict) -> str:
        """
        Genera texto según los datos proporcionados.

        Args:
            datos: Diccionario con los campos de la observación:
                - sector: Sector del proyecto
                - subsector: Subsector específico
                - aop: Actividad, Obra o Proyecto
                - texto_observacion: Texto base de la observación
                - clasificacion: Categoría de impacto
                - norma_sugerida: Normativa aplicable
                - ubicacion: Ubicación geográfica
                - tipo_accion: Acción requerida

        Returns:
            Texto formateado según el tema configurado.
        """
        return self.template.generar_completo(datos)

    def generar_desde_rag(
        self,
        texto_recuperado: str,
        datos_contexto: Dict,
        confianza: float
    ) -> str:
        """
        Genera texto integrando el resultado del RAG con el contexto.

        Args:
            texto_recuperado: Texto recuperado del Gold Standard
            datos_contexto: Datos de contexto de la observación
            confianza: Score de confianza del RAG (0-1)

        Returns:
            Texto integrado y formateado.
        """
        datos = datos_contexto.copy()

        # Integrar texto recuperado como base
        datos['texto_observacion'] = texto_recuperado

        # Agregar metadata de confianza si es tema detallado
        if self.config.tipo_redaccion == TipoRedaccion.DETALLADO:
            datos['metadata_rag'] = f"[Confianza RAG: {confianza:.2%}]"

        return self.generar(datos)

    def aplicar_formato_normativo(self, texto: str, norma: str) -> str:
        """
        Aplica formato normativo boliviano al texto.

        Args:
            texto: Texto base
            norma: Normativa a referenciar

        Returns:
            Texto con referencias normativas formateadas.
        """
        # Formatear referencias a leyes
        texto = re.sub(
            r'\bLey (\d+)\b',
            r'Ley N° \1',
            texto
        )

        # Formatear decretos supremos
        texto = re.sub(
            r'\bD\.?S\.? ?(\d+)\b',
            r'Decreto Supremo N° \1',
            texto
        )

        return texto

    def cambiar_tema(self, nuevo_config: ThemeConfig):
        """Cambia la configuración del tema dinámicamente."""
        self.config = nuevo_config
        self.template = self._seleccionar_template()

    def get_tokens_prompt(self) -> List[str]:
        """Retorna los tokens especiales para prompt engineering."""
        return self.config.tokens_contexto


# ==============================================================================
# INTEGRACIÓN CON PIPELINE RAFAELA
# ==============================================================================

class RAFAELATextPipeline:
    """
    Pipeline completo de generación de texto para RAFAELA.

    Integra la clasificación, RAG y generación de texto
    en un flujo unificado.
    """

    def __init__(self, tema: str = "oficial"):
        """
        Inicializa el pipeline con un tema predefinido.

        Args:
            tema: Nombre del tema ('oficial', 'tecnico', 'ejecutivo',
                  'consulta_publica', 'detallado')
        """
        temas_disponibles = {
            "oficial": TemasRAFAELA.tema_oficial,
            "tecnico": TemasRAFAELA.tema_tecnico,
            "ejecutivo": TemasRAFAELA.tema_ejecutivo,
            "consulta_publica": TemasRAFAELA.tema_consulta_publica,
            "detallado": TemasRAFAELA.tema_detallado
        }

        config_factory = temas_disponibles.get(tema, TemasRAFAELA.tema_oficial)
        self.generator = TextGenerator(config_factory())
        self.tema_actual = tema

    def procesar_observacion(
        self,
        datos_entrada: Dict,
        texto_gold: str,
        clasificacion_predicha: str,
        norma_predicha: str,
        score_confianza: float
    ) -> Dict[str, Any]:
        """
        Procesa una observación completa generando texto estructurado.

        Args:
            datos_entrada: Datos originales de la observación
            texto_gold: Texto recuperado del Gold Standard
            clasificacion_predicha: Clasificación del modelo
            norma_predicha: Normativa sugerida
            score_confianza: Confianza del match RAG

        Returns:
            Diccionario con todos los campos generados.
        """
        # Construir datos para generación
        datos_generacion = {
            'sector': datos_entrada.get('Sector', datos_entrada.get('SECTOR', '')),
            'subsector': datos_entrada.get('Subsector', datos_entrada.get('Subsector (Automatico)', '')),
            'aop': datos_entrada.get('Descripcion_AOP', datos_entrada.get('AOP', '')),
            'texto_observacion': texto_gold,
            'clasificacion': clasificacion_predicha,
            'norma_sugerida': norma_predicha,
            'ubicacion': datos_entrada.get('U_Ubicacion_Ref', datos_entrada.get('Ubicacion', '')),
            'tipo_accion': self._inferir_accion(clasificacion_predicha)
        }

        # Generar texto
        texto_generado = self.generator.generar_desde_rag(
            texto_gold,
            datos_generacion,
            score_confianza
        )

        return {
            'RAFAELA_Propuesta_Redaccion': texto_generado,
            'RAFAELA_Clase': clasificacion_predicha,
            'RAFAELA_Norma': norma_predicha,
            'RAFAELA_Confianza': score_confianza,
            'RAFAELA_Tema': self.tema_actual,
            'RAFAELA_Version': self.generator.config.version
        }

    def _inferir_accion(self, clasificacion: str) -> str:
        """Infiere la acción requerida según la clasificación."""
        acciones = {
            'FISICO': 'complementar el análisis del medio físico',
            'BIOTICO': 'profundizar la caracterización de flora y fauna',
            'SOCIAL': 'ampliar la consulta y participación social',
            'NORMATIVA': 'demostrar cumplimiento normativo',
            'GESTION_OPERATIVA': 'fortalecer los procedimientos de gestión',
            'DEFICIENCIA_TECNICA': 'subsanar las deficiencias técnicas identificadas'
        }
        return acciones.get(clasificacion, 'atender la observación formulada')

    def generar_lote(
        self,
        observaciones: List[Dict],
        resultados_rag: List[Tuple[str, str, str, float]]
    ) -> List[Dict]:
        """
        Procesa un lote de observaciones.

        Args:
            observaciones: Lista de datos de entrada
            resultados_rag: Lista de tuplas (texto_gold, clasificacion, norma, confianza)

        Returns:
            Lista de diccionarios con resultados procesados.
        """
        resultados = []
        for obs, (texto, clase, norma, conf) in zip(observaciones, resultados_rag):
            resultado = self.procesar_observacion(obs, texto, clase, norma, conf)
            resultados.append(resultado)
        return resultados


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def obtener_tema_por_contexto(contexto: str) -> ThemeConfig:
    """
    Selecciona automáticamente el tema según el contexto de uso.

    Args:
        contexto: Descripción del contexto ('informe_oficial',
                 'revision_tecnica', 'consulta', 'resumen')

    Returns:
        Configuración de tema apropiada.
    """
    mapeo = {
        'informe_oficial': TemasRAFAELA.tema_oficial,
        'documento_oficial': TemasRAFAELA.tema_oficial,
        'revision_tecnica': TemasRAFAELA.tema_tecnico,
        'analisis_tecnico': TemasRAFAELA.tema_tecnico,
        'consulta': TemasRAFAELA.tema_consulta_publica,
        'participacion': TemasRAFAELA.tema_consulta_publica,
        'resumen': TemasRAFAELA.tema_ejecutivo,
        'ejecutivo': TemasRAFAELA.tema_ejecutivo,
        'completo': TemasRAFAELA.tema_detallado,
        'detallado': TemasRAFAELA.tema_detallado
    }

    for clave, factory in mapeo.items():
        if clave in contexto.lower():
            return factory()

    return ThemeConfig()


def listar_temas_disponibles() -> List[str]:
    """Retorna la lista de temas predefinidos disponibles."""
    return ['oficial', 'tecnico', 'ejecutivo', 'consulta_publica', 'detallado']


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Ejemplo de uso del generador de texto

    print("=" * 70)
    print("RAFAELA - Demostración del Generador de Texto con Temas")
    print("=" * 70)

    # Datos de ejemplo
    datos_ejemplo = {
        'sector': 'HIDROCARBUROS',
        'subsector': 'Exploración',
        'aop': 'Prospección Sísmica 2D',
        'texto_observacion': """Se observa que el EsIA no presenta un análisis
        detallado de los cuerpos de agua superficiales en el área de influencia
        directa del proyecto, lo cual es fundamental para evaluar los potenciales
        impactos sobre el recurso hídrico.""",
        'clasificacion': 'FISICO',
        'norma_sugerida': 'Ley 1333, Art. 20; D.S. 3549, Art. 15',
        'ubicacion': 'Área de Influencia Directa - Cuenca del Río Grande',
        'tipo_accion': 'complementar el análisis hidrológico'
    }

    # Probar diferentes temas
    print("\n" + "-" * 70)
    print("TEMA: OFICIAL (Para Autoridad Ambiental)")
    print("-" * 70)
    pipeline_oficial = RAFAELATextPipeline(tema="oficial")
    resultado = pipeline_oficial.generator.generar(datos_ejemplo)
    print(resultado)

    print("\n" + "-" * 70)
    print("TEMA: EJECUTIVO (Resumen)")
    print("-" * 70)
    pipeline_ejecutivo = RAFAELATextPipeline(tema="ejecutivo")
    resultado = pipeline_ejecutivo.generator.generar(datos_ejemplo)
    print(resultado)

    print("\n" + "-" * 70)
    print("TEMA: CONSULTA PÚBLICA (Lenguaje Ciudadano)")
    print("-" * 70)
    pipeline_ciudadano = RAFAELATextPipeline(tema="consulta_publica")
    resultado = pipeline_ciudadano.generator.generar(datos_ejemplo)
    print(resultado)

    print("\n" + "=" * 70)
    print("Temas disponibles:", listar_temas_disponibles())
    print("=" * 70)
