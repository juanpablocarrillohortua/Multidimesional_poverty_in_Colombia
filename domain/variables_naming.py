"""
ipm_ecv2025.py
==============

Estructuras de configuración para predecir pobreza multidimensional con la
ECV 2025 (DANE) sin fuga de información entre target y features.

Contenido
---------
1. ``VARIABLES``  : registro maestro  código ECV -> pregunta, archivo, nivel.
2. ``FLUJO``      : variable base -> preguntas que solo existen según su respuesta
                    (fuga por patrón de salto).
3. ``Indicador`` / ``IndiceAF`` : especificación Alkire-Foster (Z, w, k).
4. ``IPM_DANE``   : IPM oficial de Colombia (5 dimensiones, 15 indicadores, k = 1/3).
5. ``INDICE_TIPO_PAPER`` : réplica en ECV del índice de Zhao, Cheah & Moses (2024)
                    (4 dimensiones, 11 indicadores, pesos 1/4, 1/12, 1/8, 1/20, k = 0.25).
6. Esquemas de pesos, grillas de k y conjuntos de indicadores para las pruebas
   de robustez del paper (Tablas 3, 4-6 y Fig. 2).
7. ``particion(indice)`` : devuelve los tres diccionarios
   (corte / features / excluidas con motivo) para el índice que se elija.

Las reglas del IPM oficial salen del "Algoritmo para la construcción del IPM"
(DANE). Los códigos de categoría salen del diccionario ECV 2025 (ANDA 905).
Todo lo marcado ``# [VERIFICAR]`` está en la lista ``VERIFICAR`` al final.

Ejecución: ``python ipm_ecv2025.py`` valida la configuración y exporta
``diccionarios_ipm_ecv2025.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

# =============================================================================
# 0. ARCHIVOS, LLAVES Y CONSTANTES
# =============================================================================

ARCHIVOS: Dict[str, str] = {
    "F1": "Datos de la vivienda",
    "F2": "Servicios del hogar",
    "F3": "Tenencia y financiación de la vivienda que ocupa el hogar",
    "F6": "Condiciones de vida del hogar y tenencia de bienes (Programas)",
    "F9": "Características y composición del hogar",
    "F15": "Tecnologías de información y comunicación",
    "F16": "Trabajo infantil",
    "F23": "Atención integral de los niños y niñas menores de 5 años",
    "F24": "Educación",
    "F30": "Variables diseño muestral ENCV 2025",
    "F43": "Salud",
    "F47": "Fuerza de trabajo",
    "F51": "Condiciones de vida del hogar y tenencia de bienes",
}

NIVEL_ARCHIVO: Dict[str, str] = {
    "F1": "vivienda", "F2": "hogar", "F3": "hogar", "F6": "hogar", "F51": "hogar",
    "F9": "persona", "F15": "persona", "F16": "persona", "F23": "persona",
    "F24": "persona", "F43": "persona", "F47": "persona", "F30": "persona",
}

LLAVES: Dict[str, Tuple[str, ...]] = {
    "vivienda": ("DIRECTORIO",),
    "hogar": ("DIRECTORIO", "SECUENCIA_P"),
    "persona": ("DIRECTORIO", "SECUENCIA_P", "ORDEN"),
}
# F24 usa minúsculas en algunas llaves ("Directorio", "Secuencia_p", "Orden",
# "Fex_c"): normaliza con df.columns = df.columns.str.upper() antes de cruzar.

JEFE_HOGAR = {"variable": "P6051", "archivo": "F9", "valor": 1}
NO_FAMILIA_P6051 = {10, 11, 12, 13}  # servicio doméstico, sus parientes, trabajador, pensionista

URBANO_CLASE = {1}        # CLASE: 1 = Cabecera
RURAL_CLASE = {2, 3}      # 2 = Centros poblados / 3 = Rural disperso (en 2025 solo aparece 2)

SI, NO, NS_NI = 1, 2, 9   # codificación estándar Sí/No/No sabe-No informa  # [VERIFICAR] por variable

# Líneas de pobreza monetaria 2025 (DANE, pesos corrientes, per cápita mensual, nacional).
# Se calculan con la GEIH: el ingreso de la ECV NO es la medida oficial de pobreza
# monetaria, así que aquí son solo un umbral de referencia para el índice tipo paper.
LINEA_POBREZA_MONETARIA_2025 = 482_041
LINEA_POBREZA_EXTREMA_2025 = 236_580


# =============================================================================
# 1. REGISTRO MAESTRO DE VARIABLES  (código -> pregunta)
# =============================================================================

VARIABLES: Dict[str, Dict[str, str]] = {}


def _v(codigo: str, archivo: str, pregunta: str) -> None:
    """Registra una variable. Si el código ya existe (llaves, FEX_C) no lo pisa."""
    VARIABLES.setdefault(
        codigo,
        {"pregunta": pregunta, "archivo": archivo, "nivel": NIVEL_ARCHIVO[archivo]},
    )


def _fam(prefijo: str, archivo: str, base: str, items: Dict[str, str]) -> None:
    """Registra una familia de sub-preguntas: prefijo + sufijo -> 'base: etiqueta'."""
    for sufijo, etiqueta in items.items():
        _v(prefijo + sufijo, archivo, f"{base}: {etiqueta}")


# ---- Llaves y pesos (presentes en todos los archivos) ------------------------
_v("DIRECTORIO", "F30", "Identificador de la vivienda")
_v("SECUENCIA_P", "F30", "Identificador del hogar dentro de la vivienda")
_v("SECUENCIA_ENCUESTA", "F9", "Secuencia de la encuesta (etiqueta inconsistente entre archivos)")
_v("ORDEN", "F30", "Identificador de la persona dentro del hogar")
_v("FEX_C", "F30", "Factor de expansión calibrado dentro de los bloques")

# ---- F30 Diseño muestral ----------------------------------------------------
_v("MPIO", "F30", "Identificador del municipio, 5 dígitos")
_v("SEGMENTO", "F30", "Código asociado a la MT (segmento) seleccionada, 5 dígitos")
_v("ESTRATO2020", "F30", "Estrato de muestreo")
_v("F2", "F30", "Factor de expansión de segmento a estrato de muestreo")
_v("F_ETAPA1", "F30", "Factor de expansión que expande la muestra al municipio")
_v("FRES", "F30", "Factor básico de expansión que restituye la población con la muestra")
_v("BLOQUE", "F30", "Partición geográfica del modelo de calibración")

# ---- F1 Datos de la vivienda -----------------------------------------------
_v("P1_DEPARTAMENTO", "F1", "Código del departamento")
_v("P1_MUNICIPIO", "F1", "Código del municipio")
_v("REGION", "F1", "Región")
_v("CANT_HOG_COMPLETOS", "F1", "Cantidad de hogares completos")
_v("CANT_HOGARES_VIVIENDA", "F1", "Cantidad de hogares en la vivienda")
_v("CLASE", "F1", "Clase (1 Cabecera, 2 Centros poblados, 3 Rural disperso)")
_v("P2102", "F1", "1. La vía de acceso a la edificación es:")
_v("P3155", "F1", "2. ¿Cuál es el estado de la vía?")
_v("P3156", "F1", "3. ¿La edificación donde está ubicada la vivienda tiene andén?")
_v("P1070", "F1", "4. Tipo de vivienda")
_v("P4005", "F1", "5. Material predominante de las paredes exteriores")
_v("P4015", "F1", "6. Material predominante de los pisos")
_v("P4567", "F1", "7. ¿Cuál es el material predominante del techo o cubierta?")
_v("P8520S1", "F1", "8. Servicios de la vivienda: 1. Energía eléctrica")
_v("P8520S1A1", "F1", "8. Servicios de la vivienda: Estrato para tarifa")
_v("P8520S5", "F1", "8. Servicios de la vivienda: 2. Acueducto")
_v("P8520S3", "F1", "8. Servicios de la vivienda: 3. Alcantarillado")
_v("P8520S4", "F1", "8. Servicios de la vivienda: 4. Recolección de basuras")
_v("P8520S4A1", "F1", "8. Recolección de basuras: veces por semana")
_fam("P4065", "F1", "9. En los últimos 12 meses la vivienda ha sido afectada por", {
    "S1": "Inundaciones, desbordamiento, crecientes, arroyos",
    "S2": "Avalanchas, derrumbes o deslizamientos",
    "S3": "Hundimiento del terreno",
    "S4": "Ventarrones, tormentas, vendavales, granizada",
    "S5": "Sismo o temblor",
})
_fam("P5661", "F1", "10. Frecuencia de problemas en el sector de la vivienda", {
    "S1": "Ruidos molestos del exterior", "S2": "Malos olores del exterior",
    "S3": "Basuras en calles y espacios públicos", "S4": "Contaminación del aire",
    "S9": "Contaminación en ríos, canales, lagos y embalses",
    "S5": "Invasión del espacio público", "S6": "Animales que causan molestias",
    "S7": "Insectos, roedores",
})
_fam("P3157", "F1", "11. Problemas de humedad en techos o paredes", {
    "S1": "Pintura soplada", "S2": "Presencia de manchas", "S3": "Presencia de moho",
    "S4": "Cambios en el aspecto de ladrillos, baldosas, concreto",
})
_v("P3158", "F1", "12. ¿Los techos de la vivienda presentan goteras?")

# ---- F2 Servicios del hogar ------------------------------------------------
_v("P5000", "F2", "1. Incluyendo sala y comedor, ¿de cuántos cuartos o piezas dispone este hogar?")
_v("P5010", "F2", "2. ¿En cuántos de esos cuartos duermen las personas de este hogar?")
_v("P791", "F2", "3. ¿Pagan en este hogar por la electricidad que consumen?")
_v("P5018", "F2", "4. ¿Cuánto pagaron el mes pasado o la última vez por la electricidad?")
_v("P5018S1", "F2", "4. Electricidad: ¿a cuántos meses corresponde ese pago?")
_v("P9003", "F2", "5. ¿El servicio de energía llega al hogar 24 horas, 7 días a la semana?")
_v("P9004", "F2", "6. Cuando llega el servicio de energía, ¿cuántas horas al día lo recibe?")
_v("P5666", "F2", "7. ¿En este hogar tienen servicio de gas natural conectado a red pública?")
_v("P3162", "F2", "8. ¿Pagan en este hogar por el servicio de gas natural?")
_v("P3163", "F2", "9. ¿Cuánto pagaron el mes pasado por el servicio de gas natural?")
_v("P3163S1", "F2", "9. Gas natural: ¿a cuántos meses corresponde ese pago?")
_v("P8526", "F2", "10. ¿Con qué tipo de servicio sanitario cuenta el hogar?")
_v("P3164", "F2", "11. ¿Quién realiza generalmente el mantenimiento o vaciado del servicio sanitario?")
_v("P5022", "F2", "12. El servicio sanitario está ubicado:")
_v("P5024", "F2", "13. ¿De cuántos servicios sanitarios o inodoros dispone este hogar?")
_v("P5030", "F2", "14. El servicio sanitario del hogar es: (exclusivo / compartido)")
_v("P5032", "F2", "15. ¿Pagan en este hogar por servicio de alcantarillado?")
_v("P5034", "F2", "16. ¿Cuánto pagaron el mes pasado por servicio de alcantarillado?")
_v("P5034S1", "F2", "16. Alcantarillado: ¿a cuántos meses corresponde ese pago?")
_fam("P1892", "F2", "17. El hogar cuenta con", {
    "S1": "Lavamanos", "S2": "Lavadero", "S3": "Lavaplatos", "S4": "Ninguno de los anteriores",
})
_v("P3165", "F2", "18. Para la limpieza de las manos, los miembros de este hogar principalmente:")
_v("P5041", "F2", "19. ¿Cómo eliminan principalmente la basura en este hogar?")
_v("P5038", "F2", "20. ¿Pagan en este hogar por servicio de recolección de basuras?")
_v("P5044", "F2", "21. ¿Cuánto pagaron el mes pasado por recolección de basuras?")
_v("P5044S1", "F2", "21. Basuras: ¿a cuántos meses corresponde ese pago?")
_v("P5046", "F2", "22. ¿En este hogar clasifican las basuras?")
_fam("P5046S1", "F2", "22. Tipo de material que clasifican", {
    "A1": "Desechos orgánicos", "A2": "Vidrio", "A3": "Papel y cartón", "A4": "Plástico",
    "A5": "Pilas y baterías", "A6": "Envases metálicos", "A7": "Medicamentos",
})
_fam("P5012", "F2", "23. Prácticas para reducir el consumo de agua y energía", {
    "S1": "Bombillas de bajo consumo", "S2": "Apagar luces", "S3": "Planchar en una sola ocasión",
    "S8": "Desconectar aparatos", "S4": "Reutilizar agua", "S5": "Recolectar agua lluvia",
    "S6": "Tanque sanitario de bajo consumo", "S7": "Economizadores de agua",
})
_v("P792", "F2", "24. ¿Pagan en este hogar por servicio de acueducto?")
_v("P5067", "F2", "25. ¿Cuánto pagaron el mes pasado por el servicio de acueducto?")
_v("P5067S1", "F2", "25. Acueducto: ¿a cuántos meses corresponde ese pago?")
_v("P8530", "F2", "26. El agua para preparar los alimentos la obtienen principalmente de:")
_v("P5047", "F2", "27. ¿El agua llega al hogar las 24 horas, los siete días de la semana?")
_v("P3177", "F2", "28. Cuando llega el servicio de agua, ¿cuántas horas al día lo recibe?")
_v("P3166", "F2", "29. ¿Considera que el agua disponible es suficiente?")
_v("P3167", "F2", "30. ¿Dónde está ubicado el suministro de agua (llave, tanque, pozo, etc.)?")
_v("P3168", "F2", "31. ¿Cuánto tiempo gasta caminando para recoger el agua y regresar?")
_v("P5069", "F2", "32. El agua para beber principalmente: (tratamiento)")
_v("P764", "F2", "33. ¿En dónde preparan los alimentos las personas de este hogar?")
_v("P8534", "F2", "33A. ¿El hogar tiene cocina? (espacio exclusivo para preparar alimentos)")
_fam("P3169", "F2", "34. Espacios con ventilación natural", {
    "S1": "Sala-comedor", "S2": "Cuartos para dormir", "S3": "Baños", "S4": "Cocina",
})
_v("P8536", "F2", "35. ¿Qué energía o combustible utilizan principalmente para cocinar?")
_v("P1698", "F2", "36. Adicionalmente, ¿utilizan otra fuente de energía o combustible para cocinar?")
_v("P1698S1", "F2", "36A. ¿Qué tipo de combustible (adicional) utilizan?")
_v("P8540", "F2", "37. ¿Cuánto pagaron el mes pasado en combustible para cocinar?")
_v("P5305", "F2", "38. ¿En este hogar tienen servicio telefónico fijo?")
_v("P3171", "F2", "39. ¿Este hogar realiza alguna actividad productiva en la vivienda o el lote?")
_fam("P3172", "F2", "40. Actividad productiva que realiza", {
    "S1": "Comercio", "S2": "Industrial", "S3": "Servicios", "S4": "Agropecuaria",
})
_v("P3173", "F2", "41. En los últimos 12 meses, ¿ha sufrido eventos climáticos que afectaron al hogar?")
_fam("P3174", "F2", "42. ¿Cómo lo afectaron los eventos climáticos?", {
    "S1": "Redujeron la disponibilidad de agua", "S2": "Enfermedades tropicales",
    "S3": "Enfermedades respiratorias", "S4": "Pérdida de cultivos o animales",
    "S5": "Erosión o pérdida de calidad del suelo",
})
_v("I_HOGAR", "F2", "Ingreso mensual total del hogar")
_v("I_UGASTO", "F2", "Ingreso mensual total de la unidad de gasto")
_v("PERCAPITA", "F2", "Ingreso per cápita")
_v("I_OU", "F2", "Ingreso mensual total de la otra unidad")
_v("CANT_PERSONAS_HOGAR", "F2", "43. ¿Cuántas personas componen este hogar?")

# ---- F3 Tenencia y financiación de la vivienda -----------------------------
_v("P5095", "F3", "1. La vivienda ocupada por este hogar es:")
_v("P5100", "F3", "2. ¿Cuánto pagan mensualmente por cuota de amortización?")
_v("P3197", "F3", "3. ¿Algún miembro de este hogar tiene seguros de vivienda?")
_v("P5120", "F3", "4. ¿Algún miembro de este hogar tiene escritura registrada de esta vivienda?")
_v("P8692", "F3", "5. ¿La escritura está a nombre de?")
_v("P8692S1A1", "F3", "5. Escritura: número de orden")
_v("P8692S2", "F3", "5. Escritura: ¿cuántas?")
_v("P5610", "F3", "6. El año pasado, ¿cuánto pagó por impuesto predial de esta vivienda?")
_v("P5610S1", "F3", "6. Predial: ¿a cuántos años corresponde el pago?")
_v("P8693", "F3", "7. El año pasado, ¿cuánto pagó por impuesto de valorización?")
_v("P5110", "F3", "8. Si quisiera vender esta vivienda, ¿cuál sería el precio mínimo?")
_v("P5130", "F3", "9. Si tuviera que pagar arriendo por esta vivienda, ¿cuánto estima que pagaría?")
_v("P5140", "F3", "10. ¿Cuánto pagan mensualmente por arriendo?")
_v("P3006", "F3", "11. ¿Qué tipo de contrato de arriendo tiene, verbal o escrito?")
_v("P5650", "F3", "12. ¿Cuánto pagan mensualmente por cuota de administración o celaduría?")
_v("P5160", "F3", "13. En los últimos 12 meses, ¿algún miembro recibió subsidio de vivienda?")
_v("P5160S1", "F3", "13. Subsidio de vivienda: ¿en dinero?")
_v("P5160S1A1", "F3", "13. Subsidio de vivienda en dinero: valor")
_v("P5160S2", "F3", "13. Subsidio de vivienda: ¿en especie?")
_v("P5160S2A1", "F3", "13. Subsidio de vivienda en especie: valor estimado")

# ---- F6 Programas ----------------------------------------------------------
_v("P784S4A3", "F6", "Otros programas: ¿cuántos miembros del hogar recibieron el subsidio?")
_v("P784S4A4", "F6", "Otros programas: ¿cuánto recibieron en los últimos 12 meses?")

# ---- F9 Características y composición del hogar ----------------------------
_v("P6016", "F9", "Número de orden de la persona que proporciona la información")
_v("P1894", "F9", "1. Tipo de documento de identidad")
_v("P6020", "F9", "2. Sexo al nacer")
_v("P6034", "F9", "3. ¿Cuál es la fecha de nacimiento de ...?")
_v("P6040", "F9", "4. ¿Cuántos años cumplidos tiene ...?")
_v("P6051", "F9", "5. ¿Cuál es el parentesco de ... con el jefe o la jefa de este hogar?")
_v("P5502", "F9", "6. Actualmente ...: (estado civil)")
_v("P6071", "F9", "7. El/la cónyuge de ... ¿vive en este hogar?")
_v("P756", "F9", "8. ¿Dónde nació ...?")
_v("P756S3", "F9", "8. Lugar de nacimiento: en otro país")
_v("P3510", "F9", "8A. ¿Qué nacionalidad tiene?")
_v("P3510S1A3", "F9", "8A. Nacionalidad adicional a la colombiana: venezolana")
_v("P3510S2A3", "F9", "8A. Nacionalidad (sin colombiana): Venezuela")
_v("P6074", "F9", "9. ¿... siempre ha vivido aquí en este municipio?")
_v("P755", "F9", "10. ¿Dónde vivía ... hace 5 años?")
_v("P754", "F9", "11. El lugar donde vivía ... hace 5 años era:")
_v("P753", "F9", "12. ¿Dónde vivía ... hace 12 meses?")
_v("P752", "F9", "12A. El lugar donde vivía ... hace 12 meses era:")
_v("P1662", "F9", "13. Principal motivo por el que cambió de residencia en los últimos 12 meses")
_v("P6081", "F9", "14. El padre de ... ¿vive en este hogar?")
_v("P6087", "F9", "15. ¿Cuál es o fue el nivel de educación más alto alcanzado por el padre de ...?")
_v("P6083", "F9", "16. La madre de ... ¿vive en este hogar?")
_v("P6088", "F9", "17. ¿Cuál es o fue el nivel de educación más alto alcanzado por la madre de ...?")
_v("P6080", "F9", "18. De acuerdo con su cultura, pueblo o rasgos físicos, ... es o se reconoce como:")
_v("P2057", "F9", "20. ¿Usted se considera campesino/a?")
_v("P2059", "F9", "21. ¿Usted considera que alguna vez fue campesino/a?")
_v("P2061", "F9", "22. ¿Usted considera que la comunidad en que vive es campesina?")
_v("P1895", "F9", "23. ¿Qué tan satisfecho/a se siente ... con su vida actualmente?")
_v("P1896", "F9", "24. ¿Qué tan satisfecho/a se siente ... con su ingreso actualmente?")
_v("P1897", "F9", "25. ¿Qué tan satisfecho/a se siente ... con su salud actualmente?")
_v("P1898", "F9", "26. ¿Qué tan satisfecho/a se siente ... con su nivel de seguridad?")
_v("P1899", "F9", "27. ¿Qué tan satisfecho/a se siente ... con su trabajo/actividad?")
_v("P3175", "F9", "28. ¿Qué tan satisfecho/a se siente ... con su tiempo libre?")
_v("P1901", "F9", "29. ¿Qué tan feliz se sintió ... el día de ayer?")
_v("P1903", "F9", "30. ¿Qué tan preocupado/a se sintió ... el día de ayer?")
_v("P1904", "F9", "31. ¿Qué tan triste se sintió ... el día de ayer?")
_v("P1905", "F9", "32. ¿Qué tanto considera ... que las cosas que hace en su vida valen la pena?")
_v("P1927", "F9", "33. ¿En cuál escalón diría usted que se encuentra parado/a en este momento?")
_v("P3038", "F9", "34. ¿Usted siente atracción sexual o romántica por?")
_v("P3039", "F9", "35. ¿Usted se reconoce como?")

# ---- F15 TIC ---------------------------------------------------------------
_v("P1910", "F15", "1A. ¿Con qué frecuencia utiliza ... computador de escritorio?")
_v("P1911", "F15", "1B. ¿Con qué frecuencia utiliza ... computador portátil?")
_v("P1912", "F15", "1C. ¿Con qué frecuencia utiliza ... tableta?")
_v("P1084", "F15", "2. ¿Con qué frecuencia utiliza ... internet?")
_fam("P1710", "F15", "3. Actividades que sabe hacer en computador o internet", {
    "S1": "Procesador de texto", "S2": "Copiar y pegar", "S3": "Correo con adjuntos",
    "S4": "Instalar dispositivos", "S5": "Fórmulas en hoja de cálculo", "S6": "Presentaciones",
    "S7": "Transferir archivos", "S8": "Instalar software", "S9": "Lenguaje de programación",
    "S10": "Seguridad de cuentas", "S11": "Limitar difusión de datos personales",
    "S12": "Verificar noticias",
})
_fam("P765", "F15", "4. Dispositivos que utiliza para acceder a internet", {
    "S1": "Computador de escritorio", "S2": "Portátil", "S3": "Tableta", "S4": "Celular",
    "S5": "Consola", "S6": "Televisor inteligente", "S7": "Reproductores digitales", "S8": "Otro",
})
_fam("P1085", "F15", "5. Sitios donde accede a internet", {
    "S1": "Hogar", "S2": "Trabajo", "S3": "Institución educativa", "S4": "Acceso público gratis",
    "S5": "Café internet", "S6": "Vivienda de otra persona", "S7": "En desplazamiento",
    "S8": "Otro sitio",
})
_fam("P1083", "F15", "6. Usos de internet", {
    "S1": "Obtener información", "S2": "Correo", "S3": "Redes sociales", "S4": "Comprar",
    "S5": "Banca electrónica", "S6": "Educación", "S7": "Trámites de gobierno",
    "S8": "Descargas o juegos", "S9": "Medios de comunicación", "S10": "Ver contenido audiovisual",
    "S12": "Buscar trabajo", "S13": "Servicios en la nube", "S14": "Vender productos",
    "S15": "Requisito laboral", "S16": "Llamadas por internet", "S17": "Opinar sobre productos",
    "S18": "Subir contenido propio", "S19": "Consultas educativas", "S20": "Gestiones médicas",
    "S21": "Herramientas de inteligencia artificial", "S11": "Otro",
})
_v("P1929", "F15", "7. ¿Cuál es la principal razón por la que ... no utiliza internet?")
_v("P1082", "F15", "8. ¿... tiene teléfono celular?")
_v("P1082S1", "F15", "8. Tiene celular convencional")
_v("P1082S2", "F15", "8. Tiene celular inteligente (smartphone)")
_v("P803", "F15", "9. El mes pasado, ¿... realizó pagos por telefonía celular?")
_v("P803S1", "F15", "9. Telefonía celular: valor pagado")
_v("P5504", "F15", "10. A pesar de no tener teléfono celular, ¿... utiliza teléfono celular?")
_v("P769", "F15", "12. ¿Con qué frecuencia utiliza ... teléfono celular?")
_v("P804", "F15", "14. ¿Con qué frecuencia escucha ... la señal de radio dentro del hogar?")

# ---- F16 Trabajo infantil (5 a 11 años) ------------------------------------
_v("P400", "F16", "2. (5-11 años) ¿En qué actividad ocupó la mayor parte del tiempo la semana pasada?")
_v("P401", "F16", "3. (5-11 años) ¿Realizó la semana pasada alguna actividad paga por una hora o más?")
_v("P402", "F16", "4. (5-11 años) ¿Tenía algún trabajo o negocio por el que recibe ingresos?")
_v("P403", "F16", "5. (5-11 años) ¿Trabajó en un negocio por una hora o más sin que le pagaran?")
_v("P407S2", "F16", "6. (5-11 años) Actividad de la empresa o negocio donde trabaja (código)")
_v("P5971S2", "F16", "7. (5-11 años) ¿Realizó oficios en su hogar sin que le pagaran?")
_v("P5971S2A1", "F16", "7. (5-11 años) Oficios del hogar: horas a la semana")

# ---- F23 Atención integral menores de 5 ------------------------------------
_v("P51", "F23", "1. ¿Dónde o con quién permanece ... la mayor parte del tiempo entre semana?")
_v("P772", "F23", "2. Razón principal por la que ... no asiste a hogar comunitario, jardín, CDI o colegio")
_v("P773", "F23", "3. ¿A qué tipo de establecimiento asiste?")
_v("P6169", "F23", "4. Durante este año escolar, ¿pagó matrícula para ...?")
_v("P8564", "F23", "5. ¿Pagó uniformes para ...?")
_v("P8566", "F23", "6. ¿Pagó en el establecimiento libros, útiles y elementos de aseo?")
_v("P8568", "F23", "7. ¿Compró libros y útiles por fuera del establecimiento?")
_v("P6191", "F23", "8. ¿Pensión o cuota de participación para ...?")
_v("P8572", "F23", "9. ¿Transporte para ...?")
_v("P8574", "F23", "10. ¿Al establecimiento, alimentación para ...?")
_v("P8576", "F23", "11. ¿Pagos para ... por algún otro concepto en ese establecimiento?")
_v("P55", "F23", "12. ¿Recibe o toma ... desayuno o almuerzo en el lugar donde permanece entre semana?")
_v("P774", "F23", "13. ¿... paga por esta alimentación? (desayuno o almuerzo)")
_v("P774S1", "F23", "13. Desayuno o almuerzo: valor que paga diario")
_v("P774S2", "F23", "13. Desayuno o almuerzo: valor estimado si lo comprara")
_v("P775", "F23", "14. ¿Recibe o toma ... refrigerio en el lugar donde permanece entre semana?")
_v("P776", "F23", "15. ¿... paga por esta alimentación? (refrigerio)")
_v("P776S1", "F23", "15. Refrigerio: valor que paga diario")
_v("P776S2", "F23", "15. Refrigerio: valor estimado si lo comprara")
_v("P58", "F23", "16. ¿Con quién permanece ... después de asistir o la mayor parte del tiempo?")
_v("P58S1", "F23", "16. Cuidador: número de orden")
_v("P58S2", "F23", "16. Cuidador: sexo")
_v("P777", "F23", "17. ¿Cuántos años tiene esta persona? (cuidador)")
_v("P778", "F23", "18. ¿Cuál es el nivel educativo de esta persona? (cuidador)")
_fam("P779", "F23", "19. Actividades que realiza el cuidador con el niño", {
    "S1": "Leer libros", "S2": "Contar cuentos", "S3": "Actividades artísticas", "S11": "Cantar",
    "S12": "Tocar instrumento", "S5": "Juegos y rondas", "S6": "Salir al parque",
    "S7": "Ver televisión", "S8": "Actividad deportiva", "S13": "Juegos electrónicos",
    "S9": "Otras", "S14": "Ninguna",
})

# ---- F24 Educación (5 años y más) ------------------------------------------
_v("P6160", "F24", "1. ¿Sabe leer y escribir?")
_v("P8586", "F24", "2. ¿... actualmente estudia? (preescolar, escuela, colegio o universidad)")
_v("P6218", "F24", "3. ¿Cuál es la razón principal para que ... no estudie?")
_v("P8587", "F24", "4. ¿Cuál es el nivel educativo más alto alcanzado por ...? (no estudiantes)")
_v("P8587s1", "F24", "4. Grado o año aprobado en ese nivel")
_v("P6211", "F24", "5. ¿Cuántos años de estudios superiores ha realizado y aprobado? (no estudiantes)")
_v("P1088", "F24", "6. ¿En qué nivel está matriculado/a ...? (estudiantes)")
_v("P1088s1", "F24", "6. Grado o año que cursa")
_v("P6216", "F24", "7. ¿Cuántos años de estudios superiores ha realizado y aprobado? (estudiantes)")
_v("P5673", "F24", "8. El establecimiento donde estudia ... es: (oficial / no oficial)")
_v("P5674", "F24", "8. ¿Subsidiado?")
_v("P1101", "F24", "9. ¿En qué jornada estudia ...?")
_fam("P3336", "F24", "10. Modalidad de estudio", {
    "s1": "Presencial", "s2": "Virtual", "s3": "Alternancia", "s7": "Otro",
})
_v("P3514", "F24", "11. El establecimiento educativo se localiza en:")
_v("P6223", "F24", "11A. Este establecimiento está ubicado en: (cabecera / resto)")
_v("P4693", "F24", "12. Medio de transporte principal para ir a la institución")
_v("P6167", "F24", "13. ¿Cuántos minutos gasta para ir a la institución?")
_v("P6180", "F24", "14. ¿Recibe ... en el plantel alimentos gratis o por un pago simbólico?")
_v("P6180s1", "F24", "14a. Alimentación en el plantel: valor que paga diario")
_v("P6180s2", "F24", "14b. Alimentación en el plantel: valor estimado si la comprara")
_v("P3337", "F24", "15. ¿... tuvo comunicación con sus maestros la semana pasada?")
_fam("P3339", "F24", "16. Medios para comunicarse con los maestros", {
    "s1": "Mensaje de texto", "s2": "Correo", "s3": "Teléfono", "s4": "Aplicaciones en línea",
    "s5": "WhatsApp", "s7": "Interacción personal", "s6": "Otro",
})
for _cod, _txt in {
    "P3341": "17. Durante este año escolar el hogar pagó matrículas",
    "P3342": "18. Durante este año escolar el hogar pagó uniformes",
    "P3343": "19. Durante este año escolar el hogar pagó útiles o textos",
    "P3344": "20. El hogar paga pensión",
    "P3345": "21. El hogar paga transporte escolar",
    "P3346": "22. El hogar paga alimentación en el establecimiento",
    "P3347": "23. El mes pasado el hogar gastó en útiles o fotocopias",
    "P3348": "24. El mes pasado el hogar realizó otros pagos en el establecimiento",
}.items():
    _v(_cod, "F24", _txt)
    _v(_cod + "s1", "F24", _txt + ": valor")
_v("P8610", "F24", "25. Durante este año escolar, ¿... recibió beca para estudiar?")
_v("P8610s1", "F24", "25. Beca: valor")
_v("P6229", "F24", "26. ¿De quién recibió la beca para estudiar?")
_v("P8612", "F24", "27. Durante este año escolar, ¿... recibió subsidio para estudiar?")
_v("P8612s1", "F24", "27. Subsidio educativo: valor")
_v("P6238", "F24", "28. ¿De quién recibió el subsidio para estudiar?")
_v("P8614", "F24", "29. Durante este año escolar, ¿... ha recibido crédito educativo?")
_v("P8614s1", "F24", "29. Crédito educativo: valor")
_v("P6202", "F24", "30. ¿Quién le concedió el crédito educativo?")
_v("P781", "F24", "31. ¿Con quién permanece ... después de asistir o la mayor parte del tiempo? (5-17)")
_v("P782", "F24", "32. ¿Cuántos años tiene esta persona? (cuidador 5-17)")
_v("P783", "F24", "33. ¿Cuál es el nivel educativo de esta persona? (cuidador 5-17)")
_fam("P3004", "F24", "34. Actividades fuera de la jornada escolar", {
    "s1": "Grupos artísticos", "s2": "Ciencia y tecnología", "s3": "Escuelas deportivas",
    "s4": "Grupos de estudio", "s5": "Sale al parque", "s6": "Lee libros", "s7": "Juega",
    "s8": "Ninguna",
})

# ---- F43 Salud ---------------------------------------------------------------
_v("P6090", "F43", "1. ¿... está afiliado/a (cotizante o beneficiario) a alguna entidad de seguridad social en salud?")
_v("P768", "F43", "2. ¿Por qué razón principal no está afiliado/a?")
_v("P6100", "F43", "3. ¿A cuál régimen de seguridad social en salud está afiliado/a?")
_v("P6115", "F43", "4. ¿Quién paga mensualmente por la afiliación de ...?")
_v("P5669", "F43", "5. ¿De quién es beneficiario/a ...?")
_v("P8551", "F43", "6. ¿Cuánto paga o le descuentan mensualmente para estar cubierto?")
_v("P3511", "F43", "6A. ¿Dónde está ubicada la entidad donde recibe los servicios de salud?")
_v("P6181", "F43", "7. Calidad del servicio de su EPS")
_v("P798", "F43", "8. Aspecto que más influye en su percepción de la calidad de la EPS")
_fam("P799", "F43", "9. Planes o seguros voluntarios de salud", {
    "S2": "Medicina prepagada", "S3": "Plan complementario", "S1": "Póliza de hospitalización",
    "S4": "Seguro estudiantil", "S5": "Otro (ambulancia, domiciliaria)",
})
_v("P3176", "F43", "10. ¿Cuánto paga mensualmente por planes o seguros voluntarios de salud?")
_v("P6127", "F43", "11. El estado de salud de ... en general, es:")
_v("P1930", "F43", "12. En los últimos 12 meses, ¿le han diagnosticado alguna enfermedad crónica?")
_v("P1930S1", "F43", "12. ¿Actualmente está recibiendo tratamiento para esa enfermedad?")
_fam("P1699", "F43", "12A. Consulta preventiva al menos una vez al año", {
    "S1": "Medicina general", "S3": "Medicina especializada", "S2": "Odontología",
    "S4": "Medicina alternativa", "S5": "Psicología",
})
_fam("P1906", "F43", "13. Dificultades en la vida diaria por condición física o mental", {
    "S1": "Oír", "S2": "Hablar", "S3": "Ver", "S4": "Moverse o caminar",
    "S5": "Agarrar objetos", "S6": "Entender o recordar", "S7": "Autocuidado",
    "S8": "Relacionarse",
})
_v("P6126", "F43", "16. ¿Quién se ocupa principalmente del cuidado de ...?")
_v("P6126S3", "F43", "16. ¿Esta persona tuvo que dejar de trabajar para dedicarse al cuidado?")
_v("P5665", "F43", "17. En los últimos 30 días, ¿tuvo alguna enfermedad, accidente o problema de salud sin hospitalización?")
_v("P6134", "F43", "18. Por ese problema de salud, ¿cuántos días dejó de realizar sus actividades?")
_v("P8563", "F43", "19. Para tratar ese problema de salud, ¿qué hizo principalmente ...?")
_v("P1092", "F43", "20. ¿Acudió al servicio de urgencias?")
_v("P8573", "F43", "21. ¿Le brindaron asistencia médica en urgencias?")
_v("P8575", "F43", "22. Razón por la que no recibió atención en urgencias")
_v("P8577", "F43", "23. Tiempo entre llegar a urgencias y ser atendido/a")
_v("P770", "F43", "24. En urgencias fue atendido/a por:")
_v("P6153", "F43", "25. ¿Cuál fue la razón principal por la que no solicitó o no recibió atención médica?")
_v("P6199", "F43", "26. Días entre pedir la cita y la consulta")
_v("P6145", "F43", "27. ¿Fue remitido/a a especialista?")
_v("P8554", "F43", "28. Calidad de la prestación del servicio de salud")
_v("P801", "F43", "29. Aspecto que más influyó en la percepción de la calidad")
_fam("P8556", "F43", "30. Fuentes para cubrir los costos de atención en los últimos 30 días", {
    "S1": "EPS", "S2": "Plan voluntario", "S4": "SOAT", "S5": "Secretaría de salud o alcaldía",
    "S6": "Recursos propios", "S9": "Recursos de otras personas", "S10": "No se requirió pago",
})
_v("P6147", "F43", "31. Por esta enfermedad, ¿le formularon medicamentos?")
_v("P6148", "F43", "32. ¿Los medicamentos le fueron entregados por la institución a la que está afiliado?")
_v("P6149", "F43", "33. ¿Por qué razón no le fueron entregados los medicamentos?")
_v("P3178", "F43", "34. Durante los últimos 30 días, pagos por atención en salud (bloque P3178-P318610)")
_v("P3008", "F43", "36. ¿Actualmente ... fuma?")
_v("P1707", "F43", "37. ¿... consume bebidas azucaradas?")
_v("P3003", "F43", "38. ¿... consume alimentos de paquete?")
_v("P6133", "F43", "39. Durante los últimos 12 meses, ¿tuvo que ser hospitalizado/a?")
_v("P3189", "F43", "41. ¿Cuánto pagó en total por esta hospitalización?")
_v("P8561", "F43", "42. Calidad del servicio en esta hospitalización")
_v("P3335", "F43", "43. ¿Ha estado embarazada alguna vez en su vida?")
_v("P3335S1", "F43", "43a. ¿Cuántos hijos nacidos vivos ha tenido?")
_v("P3335S1A1", "F43", "43b. ¿A qué edad tuvo su primer hijo?")
_v("P8584", "F43", "44. ¿... está embarazada actualmente?")
_v("P5694", "F43", "45. ¿Asiste a control prenatal?")
_v("P5452", "F43", "46. ¿Tiene el esquema completo de vacunación según su edad?")
_v("P6161", "F43", "47. ¿Llevan a ... a valoración integral o control de crecimiento y desarrollo?")
_v("P1089", "F43", "48. Razón principal para no llevar a ... a control de crecimiento y desarrollo")

# ---- F47 Fuerza de trabajo (12 años y más) ---------------------------------
_v("P6230", "F47", "1. Número de orden de la persona que suministra la información")
_v("P6240", "F47", "2. ¿En qué actividad ocupó ... la mayor parte del tiempo la semana pasada?")
_v("P6250", "F47", "3. ¿... realizó la semana pasada alguna actividad paga por una hora o más?")
_v("P6260", "F47", "4. Aunque no trabajó, ¿tenía algún trabajo o negocio por el que recibe ingresos?")
_v("P6270", "F47", "5. ¿... trabajó la semana pasada en un negocio una hora o más sin que le pagaran?")
_v("P6280", "F47", "6. En las últimas 4 semanas, ¿hizo alguna diligencia para conseguir trabajo?")
_v("P6300", "F47", "7. ¿Desea ... conseguir un trabajo remunerado o instalar un negocio?")
_v("P6320", "F47", "8. Durante los últimos 12 meses, ¿trabajó por lo menos 2 semanas consecutivas?")
_v("P6330", "F47", "9. Después de su último empleo, ¿ha hecho alguna diligencia para conseguir trabajo?")
_v("P6340", "F47", "10. Durante los últimos 12 meses, ¿ha hecho alguna diligencia para conseguir trabajo?")
_v("P6351", "F47", "11. Si le hubiera resultado algún trabajo, ¿estaba disponible la semana pasada?")
_v("P6370S2", "F47", "¿Qué hace ... en este trabajo? (ocupación, código)")
_v("P6390S2", "F47", "Actividad principal de la empresa o negocio (rama, código)")
_v("P6435", "F47", "15. En este trabajo ... es: (posición ocupacional)")
_v("P6440", "F47", "16. ¿Para realizar este trabajo tiene algún tipo de contrato?")
_v("P6450", "F47", "17. ¿Qué tipo de contrato tiene, verbal o escrito?")
_v("P6460", "F47", "18. ¿El contrato es a término indefinido o a término fijo?")
_v("P6990", "F47", "19. ¿Está afiliado a una Administradora de Riesgos Laborales (ARL)?")
_v("P6426", "F47", "20. ¿Cuántos meses lleva trabajando en esta empresa de manera continua?")
_v("P8624", "F47", "21. Antes de descuentos, ¿cuánto ganó ... el mes pasado en este empleo?")
for _cod, _txt in {
    "P6595": "22. Recibió alimentos como parte de pago", "P6605": "23. Recibió vivienda como parte de pago",
    "P6623": "24. Recibió otros ingresos en especie", "P6615": "25. Usa transporte de la empresa",
    "P8626": "26. Recibió subsidio de alimentación", "P8628": "27. Recibió auxilio de transporte",
    "P8630": "28. Recibió subsidio familiar", "P8631": "29. Recibió primas",
}.items():
    _v(_cod, "F47", _txt)
    _v(_cod + "S1", "F47", _txt + ": valor mensual")
_fam("P1087", "F47", "30. Durante los últimos 12 meses recibió", {
    "S1": "Prima de servicios", "S1A1": "Prima de servicios (valor)",
    "S2": "Prima de navidad", "S2A1": "Prima de navidad (valor)",
    "S3": "Prima de vacaciones", "S3A1": "Prima de vacaciones (valor)",
    "S4": "Bonificaciones", "S4A1": "Bonificaciones (valor)",
    "S5": "Indemnizaciones por accidente de trabajo", "S5A1": "Indemnizaciones (valor)",
})
_v("P6750", "F47", "31. Ganancia neta u honorarios netos en esa actividad el mes pasado")
_v("P550", "F47", "32. Ganancia neta del negocio o la cosecha en los últimos 12 meses")
_v("P8632", "F47", "33. ¿Cuántas personas tiene la empresa o negocio donde trabaja?")
_v("P8634", "F47", "34. ¿Dónde realiza su trabajo principal?")
_v("P3515", "F47", "35. La empresa donde trabaja está ubicada en:")
_v("P6885", "F47", "36. Medio de transporte principal para ir al trabajo")
_v("P6886", "F47", "37. Tiempo de viaje de ida al trabajo")
_v("P415", "F47", "38. ¿Cuántas horas a la semana trabaja normalmente?")
_v("P416", "F47", "39. ¿Cuántas horas trabajó la semana pasada?")
_v("P6830", "F47", "40. Razón por la que no trabajó algunas de sus horas normales")
_v("P9000", "F47", "41. ¿Desea trabajar más horas?")
_v("P9001", "F47", "42. En las últimas 4 semanas, ¿hizo diligencias para trabajar más horas?")
_v("P3193", "F47", "42A. En los últimos 12 meses, ¿ha trabajado en teletrabajo o trabajo en casa?")
_v("P8636", "F47", "43. El mes pasado, ¿tuvo otros trabajos o negocios con ingresos?")
_v("P8636S1", "F47", "43. Otros trabajos: valor total percibido")
_v("P7250", "F47", "44. ¿Durante cuántas semanas ha estado o estuvo ... buscando trabajo?")
_v("P7310", "F47", "45. ¿Busca trabajo por primera vez o había trabajado antes?")
_v("P7270S2", "F47", "46. Ocupación en la que ha buscado trabajo (código)")
_v("P8640", "F47", "47. El mes pasado, ¿recibió algún ingreso por concepto de trabajo?")
_v("P8640S1", "F47", "47. Ingreso por trabajo: valor")
_v("P6920", "F47", "48. ¿Está ... cotizando actualmente a un fondo de pensiones?")
_v("P6935", "F47", "49. ¿A cuál fondo cotiza actualmente?")
_v("P8642", "F47", "50. El mes pasado, ¿recibió ingreso por pensión de jubilación, sustitución, invalidez o vejez?")
_v("P8642S1", "F47", "50. Pensión: valor")
_v("P8644", "F47", "51. El mes pasado, ¿recibió dinero para el sostenimiento de hijos menores de 18?")
_v("P8644S1", "F47", "51. Sostenimiento de hijos: valor")
_v("P8646", "F47", "52. El mes pasado, ¿recibió ingresos por arriendos?")
_v("P8646S1", "F47", "52. Arriendos: valor")
_v("P8648", "F47", "53. En los últimos 12 meses, ¿recibió primas por pensión?")
_v("P8648S1", "F47", "53. Primas por pensión: valor")
_v("P8650", "F47", "54. En los últimos 12 meses, ¿recibió ayudas en dinero de otros hogares o instituciones?")
_v("P8650S1", "F47", "54. El dinero provino de:")
_v("P8650S1A1", "F47", "54. Ayudas: valor")
_v("P8652", "F47", "55. En los últimos 12 meses, ¿recibió dinero por venta de propiedades?")
_v("P8652S1", "F47", "55. Venta de propiedades: valor")
_v("P8654", "F47", "56. En los últimos 12 meses, ¿recibió dinero por otros conceptos (cesantías, intereses...)?")
_v("P8654S1", "F47", "56. Otros conceptos: valor")
_fam("P9002", "F47", "57. Razones por las que no está ocupado o no busca trabajo", {
    "S1": "Cuidar niños menores de 5", "S2": "Cuidar niños de 5 y más", "S3": "Cuidar persona mayor",
    "S4": "Cuidar persona con discapacidad", "S5": "Oficios del hogar", "S6": "Está estudiando",
    "S7": "Ya se pensionó", "S8": "Es adulto mayor", "S9": "No desea trabajar", "S10": "Otra",
})
_v("P421S2", "F47", "58. ¿Realizó oficios en su hogar sin que le pagaran?")
_v("P421S2A1", "F47", "58. Oficios del hogar: horas")

# ---- F51 Condiciones de vida y tenencia de bienes ---------------------------
_v("P1079", "F51", "1. ¿Quién contesta el capítulo?")
_v("P9010", "F51", "2. ¿Cómo se siente en el barrio, pueblo o vereda donde vive?")
_fam("P9025", "F51", "3. Hechos de los que fue víctima el hogar en 12 meses", {
    "S1": "Atracos o robos", "S2": "Otro hecho violento",
})
_fam("P3180", "F51", "4. Problemas que se presentaron en el barrio o vereda en 12 meses", {
    "S1": "Hurtos", "S2": "Homicidios", "S3": "Extorsiones", "S4": "Riñas", "S5": "Vandalismo",
    "S6": "Consumo de drogas", "S7": "Petardos o granadas", "S8": "Hurto de ganado",
    "S9": "Paros armados", "S10": "Cultivos ilícitos", "S11": "Minería ilegal",
    "S12": "Tala de árboles", "S13": "Contaminación hídrica", "S14": "Violaciones sexuales",
    "S15": "Acoso sexual en espacios públicos",
})
_fam("P9005", "F51", "5. Presencia de grupos en el barrio o vereda en 12 meses", {
    "S1": "Combos o pandillas", "S2": "Expendedores de droga", "S3": "Barras bravas",
    "S4": "Guerrilla", "S5": "Grupos armados organizados",
})
_v("P5230", "F51", "6. ¿Usted se considera pobre?")
_v("P9090", "F51", "7. Los ingresos de su hogar: (alcanzan / no alcanzan para los gastos mínimos)")
_fam("P784", "F51", "8. Ayudas o subsidios del gobierno en 12 meses", {
    "S2": "Colombia Mayor", "S2A1": "Colombia Mayor (beneficiarios)", "S2A2": "Colombia Mayor (monto)",
    "S7": "Renta ciudadana: Colombia sin hambre", "S7A1": "Colombia sin hambre (beneficiarios)",
    "S7A2": "Colombia sin hambre (monto)", "S6": "Renta ciudadana: devolución del IVA",
    "S6A1": "Devolución del IVA (beneficiarios)", "S6A2": "Devolución del IVA (monto)",
    "S4": "Otros programas", "S4A1": "Otros programas (cuántos)",
})
_v("P1072", "F51", "9. ¿Cuántas libras de arroz consumió este hogar la semana pasada?")
_fam("P1077", "F51", "10. Bienes o servicios que posee el hogar", {
    "S1": "Lavadora de ropa", "S2": "Nevera", "S3": "Estufa eléctrica o de gas",
    "S6": "Calentador o ducha eléctrica", "S27": "Plancha", "S10": "Equipo de sonido",
    "S28": "Suscripción a plataformas de video o música", "S7": "TV a color convencional",
    "S8": "TV LCD, plasma o LED", "S11": "Aire acondicionado", "S12": "Ventilador",
    "S14": "Consola de videojuegos", "S15": "Carro particular", "S15A1": "Carros (cantidad)",
    "S16": "Moto", "S16A1": "Motos (cantidad)", "S17": "Bicicleta", "S30": "Patineta eléctrica",
    "S18": "Casa, apartamento o finca de recreo", "S21": "Computador de escritorio",
    "S21A1": "Computadores de escritorio (cantidad)", "S22": "Computador portátil",
    "S22A1": "Portátiles (cantidad)", "S23": "Tableta", "S23A1": "Tabletas (cantidad)",
})
_v("P1076", "F51", "11. ¿Cuál es la principal razón por la que el hogar no tiene computador?")
_v("P1075", "F51", "12. ¿El hogar tiene conexión a internet?")
_v("P1075S1", "F51", "12. Conexión a internet fija")
_v("P1075S2", "F51", "12. Conexión a internet móvil")
_v("P1074", "F51", "13. ¿Cuál es la razón principal por la que el hogar no tiene internet?")
_fam("P795", "F51", "14. Servicio que usa el hogar para ver televisión", {
    "S1": "Señal abierta", "S2": "Cable", "S3": "Satelital", "S4": "IPTV",
})
_fam("P1913", "F51", "15. Tiempo caminando hasta el servicio más cercano", {
    "S1": "Paradero de transporte público", "S2": "Colegio, jardín o CDI",
    "S3": "Hospital o centro de salud", "S10": "Supermercado o tienda", "S9": "Plaza de mercado",
    "S5": "Estación de policía o CAI", "S6": "Droguería", "S7": "Parque o zonas verdes",
    "S11": "Escenarios culturales", "S12": "Espacios deportivos", "S8": "Banco o cajero",
    "S13": "Centro de venta o acopio (rural)",
})
_fam("P3202", "F51", "16. Eventos ocurridos al hogar en 12 meses", {
    "S1": "El jefe perdió su empleo", "S2": "El cónyuge perdió su empleo",
    "S3": "Otro miembro perdió su empleo", "S4": "Cerró el negocio", "S5": "No pudo vender la producción",
    "S6": "Atraso en pago de colegio o jardín", "S7": "No pudo pagar universidad o técnico",
    "S8": "Atraso en pago de vivienda", "S9": "Atraso en administración", "S10": "Atraso en servicios públicos",
    "S11": "Atraso en impuestos", "S12": "Ninguno",
})
_fam("P3203", "F51", "17. Medidas para enfrentar los eventos", {
    "S1": "Empezaron a trabajar miembros que no trabajaban", "S2": "Nuevas fuentes de ingreso",
    "S3": "Se fueron a vivir con familiares", "S4": "Gastaron ahorros", "S5": "Se endeudaron",
    "S6": "Vendieron bienes", "S7": "Vendieron la vivienda", "S8": "Retiraron hijos de la escuela",
    "S9": "Retiraron hijos de la universidad", "S10": "Disminuyeron gasto en alimentos",
    "S11": "Pidieron ayuda al gobierno", "S12": "Pidieron ayuda a familiares o vecinos",
    "S13": "Solicitaron subsidio al desempleo", "S14": "Otra",
})
_v("P3353", "F51", "18. Situación económica del hogar comparada con hace 12 meses")
_v("P3354", "F51", "19. Situación económica del hogar dentro de 12 meses")
_v("P3355", "F51", "20. Situación económica del país comparada con hace 12 meses")
_v("P3356", "F51", "21. Situación económica del país dentro de 12 meses")
_v("P3357", "F51", "22. Posibilidades de comprar ropa, zapatos, alimentos frente a hace un año")
_v("P3358", "F51", "23. Posibilidades de comprar muebles o electrodomésticos frente a hace un año")
_fam("P3516", "F51", "24. (FIES) En 12 meses, por falta de dinero, alguien del hogar", {
    "S1": "Se preocupó por no tener suficientes alimentos", "S2": "No pudo comer alimentos saludables",
    "S3": "Comió poca variedad", "S4": "Saltó una comida", "S5": "Comió menos de lo que debía",
    "S6": "El hogar se quedó sin alimentos", "S7": "Tuvo hambre pero no comió",
    "S8": "No comió en un día entero",
})


# =============================================================================
# 1. Variables IPM
# =============================================================================

variables_ipm_dane = {
    # Identificadores
    "DIRECTORIO": "Identificador de la vivienda",
    "SECUENCIA_ENCUESTA": "Número de hogar (archivos de hogar) o de persona (archivos de personas)",
    "SECUENCIA_P": "Número de hogar dentro de la vivienda (archivos de personas)",
    "ORDEN": "Número de persona dentro del hogar",
    "FEX_C": "Factor de expansión",

    # Geografía (Datos de la vivienda)
    "P1_DEPARTAMENTO": "Código del departamento (2 dígitos, leer como str)",
    "P1_MUNICIPIO": "Código del municipio (3 dígitos, leer como str)",
    "CLASE": "1 = cabecera, 2 = centros poblados y rural disperso",

    # Personas (Características y composición del hogar)
    "P6040": "Edad en años cumplidos",
    "P6020": "Sexo al nacer",
    "P6051": "Parentesco con el jefe del hogar",

    # Educación
    "P6160": "¿Sabe leer y escribir?",
    "P8586": "¿Actualmente estudia?",
    "P8587": "Nivel educativo más alto alcanzado",
    "P8587S1": "Grado o año aprobado en ese nivel",
    "P1088": "Nivel en el que está matriculado",
    "P1088S1": "Grado o año que cursa",
    "P6180": "¿Recibe alimentos en el plantel gratis o por pago simbólico?",

    # Primera infancia (Atención integral de menores de 5 años)
    "P51": "Dónde o con quién permanece la mayor parte del tiempo entre semana",
    "P55": "¿Recibe desayuno o almuerzo en el lugar donde permanece?",
    "P774": "¿Paga por esta alimentación?",

    # Salud
    "P6090": "Afiliación a seguridad social en salud",
    "P5665": "¿Tuvo un problema de salud en los últimos 30 días sin hospitalización?",
    "P8563": "¿Qué hizo principalmente para tratar ese problema de salud?",

    # Fuerza de trabajo
    "P6240": "Actividad en la que ocupó la mayor parte del tiempo la semana pasada",
    "P6250": "¿Realizó alguna actividad paga por una hora o más?",
    "P6260": "¿Tenía trabajo o negocio aunque no trabajó?",
    "P6270": "¿Trabajó en un negocio sin que le pagaran?",
    "P6280": "¿Hizo diligencias para conseguir trabajo en las últimas 4 semanas?",
    "P6351": "¿Estaba disponible la semana pasada para empezar a trabajar?",
    "P7250": "Semanas buscando trabajo (privado si son más de 52)",
    "P6920": "¿Cotiza actualmente a un fondo de pensiones?",
    "P8642": "¿Recibió el mes pasado ingreso por pensión? (condición de pensionado)",

    # Vivienda (Datos de la vivienda)
    "P8520S5": "¿Tiene acueducto?",
    "P8520S3": "¿Tiene alcantarillado?",
    "P4015": "Material predominante de los pisos",
    "P4005": "Material predominante de las paredes exteriores",

    # Servicios del hogar
    "P8530": "Fuente principal del agua para preparar alimentos",
    "P8526": "Tipo de servicio sanitario",
    "P5010": "Número de cuartos en que duermen las personas del hogar",
    "CANT_PERSONAS_HOGAR": "Número de personas que componen el hogar",
}
