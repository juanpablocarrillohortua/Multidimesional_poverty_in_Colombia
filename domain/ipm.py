"""
ipm.py
======

Índice de Pobreza Multidimensional (IPM) para Colombia con el método de
doble corte de Alkire-Foster, siguiendo la metodología del DANE y los
códigos de la Encuesta Nacional de Calidad de Vida (ECV) 2025.

Uso
---
    from domain.ipm import (
        IPM,
        INDICADORES_ELIMINADOS,
        CARACTERISTICAS_NUEVAS,
        VARIABLES_CORTE,
    )

    # IPM oficial (15 indicadores)
    hogares = IPM().fit_transform(df)

    # Esquema modificado: 9 indicadores en el corte, 6 fuera del corte
    ipm = IPM(caracteristicas=INDICADORES_ELIMINADOS)
    hogares = ipm.fit_transform(df)
    y = hogares["POBRE"]
    X_car = hogares[CARACTERISTICAS_NUEVAS]   # versión binaria (CAR_*)
    X_org = hogares[ORIGINALES_NUEVAS]        # versión en valores originales (ORG_*)
    IPM.resumen(hogares)

`df` debe tener una fila por persona, con todas las tablas de la ECV
consolidadas y los nombres de columnas en mayúsculas.

Análisis de robustez del corte (métodos estáticos, ver cada docstring)
---------------------------------------------------------------------
    w = ipm.w_

    # Figura 2: robustez al corte k
    dom = IPM.dominancia_k(hogares, w)                  # M0 de 4 grupos aleatorios
    IPM.grafico_dominancia(dom)

    # Tabla 3: robustez a los indicadores
    esq = IPM.esquemas_sin_indicador(w, ipm.indicadores_)
    tau, p, m0 = IPM.kendall_esquemas(hogares, esq)
    IPM.grafico_kendall(tau, p)

    # Tabla 6: robustez a los pesos
    esq = IPM.esquemas_peso_dimension(w, ipm.indicadores_)
    tau, p, m0 = IPM.kendall_esquemas(hogares, esq)
    IPM.grafico_kendall(tau, p)

Referencia: Zhao, Cheah y Moses (2024), PLOS ONE 19(5): e0300263.

Nota sobre sklearn: IPM convierte filas de personas en filas de hogares,
así que no puede ir antes de un modelo dentro de un mismo Pipeline. Córrelo
una vez antes de partir los datos (fit no aprende nada de X, así que no hay
fuga) y luego divide por hogar o segmento, por ejemplo con GroupKFold.
"""

import operator

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "CARACTERISTICAS_EXCLUIDAS",
    "CARACTERISTICAS_NUEVAS",
    "DANE_CODIGOS",
    "DANE_INDICADORES",
    "DANE_K",
    "INDICADORES_ELIMINADOS",
    "IPM",
    "ORIGINALES_EXCLUIDAS",
    "ORIGINALES_NUEVAS",
    "VARIABLES_COMPARTIDAS",
    "VARIABLES_CORTE",
    "VARIABLES_IPM",
    "VARIABLES_LIBERADAS",
    "pesos_equiponderados",
]


# ----------------------------------------------------------------------
# Variables de la ECV 2025 usadas en el cálculo
# ----------------------------------------------------------------------
VARIABLES_IPM = {
    # Identificadores
    "DIRECTORIO": "Identificador de la vivienda",
    "SECUENCIA_ENCUESTA": (
        "Número de hogar (archivos de hogar) o de persona (archivos de personas)"
    ),
    "SECUENCIA_P": "Número de hogar dentro de la vivienda (archivos de personas)",
    "ORDEN": "Número de persona dentro del hogar",
    "FEX_C": "Factor de expansión",
    # Geografía (Datos de la vivienda)
    "P1_DEPARTAMENTO": "Código del departamento (2 dígitos, leer como str)",
    "P1_MUNICIPIO": "Código del municipio (3 dígitos, leer como str)",
    "CLASE": "1 = cabecera, 2 = centros poblados y rural disperso",
    # Personas
    "P6040": "Edad en años cumplidos",
    "P6020": "Sexo al nacer",
    "P6051": "Parentesco con el jefe del hogar",
    # Educación
    "P6160": "¿Sabe leer y escribir?",
    "P8586": "¿Actualmente estudia?",
    "P8587": "Nivel educativo más alto alcanzado",
    "P8587S1": "Grado o año aprobado en ese nivel",
    "P6211": "Años de estudios superiores aprobados (no estudia)",
    "P1088": "Nivel en el que está matriculado",
    "P1088S1": "Grado o año que cursa",
    "P6216": "Años de estudios superiores aprobados (estudia)",
    "P6180": "¿Recibe alimentos en el plantel gratis o por pago simbólico?",
    # Primera infancia
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
    "P7250": "Semanas buscando trabajo",
    "P6920": "¿Cotiza actualmente a un fondo de pensiones? (3 = ya es pensionado)",
    "P8642": "¿Recibió el mes pasado ingreso por pensión?",
    # Vivienda
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


# ----------------------------------------------------------------------
# Parámetros DANE por defecto
# ----------------------------------------------------------------------
# Cada indicador: dimensión, filtro de personas elegibles, variable,
# operador y umbral z. "var", "op" y "z" pueden ser dict {CLASE: valor}
# cuando el corte cambia entre cabecera (1) y resto (2).
# agg: "any" (al menos un miembro) o "mean" (promedio del hogar).
# vacio: valor del hogar si no tiene miembros elegibles.
# solo_pensionados: valor forzado si el hogar es solo de pensionados.
DANE_INDICADORES = {
    "logro_educativo": dict(
        dim="educacion",
        filtro=[("P6040", ">=", 15)],
        var="ANOS_EDUC",
        agg="mean",
        op="<",
        z=9,
        vacio=1,
    ),
    "analfabetismo": dict(
        dim="educacion",
        filtro=[("P6040", ">=", 15)],
        var="P6160",
        op="==",
        z=2,
        vacio=1,
    ),
    "rezago_escolar": dict(
        dim="ninez",
        filtro=[("P6040", ">=", 7), ("P6040", "<=", 17)],
        var="BRECHA_EDUC",
        op="<",
        z=0,
        vacio=0,
    ),
    "inasistencia_escolar": dict(
        dim="ninez",
        filtro=[("P6040", ">=", 6), ("P6040", "<=", 16)],
        var="P8586",
        op="==",
        z=2,
        vacio=0,
    ),
    "primera_infancia": dict(
        dim="ninez",
        filtro=[("P6040", "<=", 5)],
        var="SIN_CUIDADO_PI",
        op="==",
        z=1,
        vacio=0,
    ),
    "trabajo_infantil": dict(
        dim="ninez",
        filtro=[("P6040", ">=", 12), ("P6040", "<=", 17)],
        var="OCUPADO",
        op="==",
        z=1,
        vacio=0,
    ),
    "sin_aseguramiento": dict(
        dim="salud",
        filtro=[("P6040", ">=", 6)],
        var="P6090",
        op="in",
        z=[2, 9],
        vacio=0,
    ),
    "barreras_salud": dict(
        dim="salud",
        filtro=[("P5665", "==", 1)],
        var="P8563",
        op="in",
        z=[3, 4, 5, 6, 7, 8],
        vacio=0,
    ),
    # La regla "más de N semanas buscando trabajo" se calcula una sola vez en
    # _preparar (columna DLD); el umbral está en DANE_CODIGOS["semanas_dld"].
    "desempleo_larga": dict(
        dim="trabajo",
        filtro=[("PEA", "==", 1)],
        var="DLD",
        op="==",
        z=1,
        vacio=1,
        solo_pensionados=1,
    ),
    "trabajo_informal": dict(
        dim="trabajo",
        filtro=[("PEA2", "==", 1)],
        var="FORMAL",
        op="==",
        z=0,
        vacio=1,
        solo_pensionados=0,
    ),
    "agua": dict(
        dim="vivienda",
        var={1: "P8520S5", 2: "P8530"},
        op={1: "==", 2: "in"},
        z={1: 2, 2: [4, 5, 6, 8, 9, 10]},
    ),
    "excretas": dict(
        dim="vivienda",
        var={1: "P8520S3", 2: "P8526"},
        op={1: "==", 2: "in"},
        z={1: 2, 2: [3, 5, 6]},
    ),
    "pisos": dict(dim="vivienda", var="P4015", op="==", z=7),
    "paredes": dict(
        dim="vivienda",
        var="P4005",
        op="in",
        z={1: [5, 7, 8, 9, 10], 2: [7, 8, 9, 10]},
    ),
    "hacinamiento": dict(
        dim="vivienda",
        var="PERSONAS_CUARTO",
        op={1: ">=", 2: ">"},
        z=3,
    ),
}

# Códigos y umbrales usados para construir variables auxiliares
DANE_CODIGOS = {
    "salud_no_afiliado": [2, 9],  # P6090: no / no sabe
    "pi_asiste": [1],  # P51: hogar comunitario, jardín, CDI o colegio
    "pi_cuidado_inadecuado": [
        3,
        6,
        7,
    ],  # P51: padre/madre en trabajo, pariente <18, solo
    "pi_alimento_ok": [2, 3],  # P774: pago simbólico o gratis
    "semanas_dld": 52,  # P7250: desempleo de larga duración si es mayor
    "tope_anos_superior": 10,  # P6211 / P6216: tope de años de educación superior
}

DANE_K = 5 / 15


# ----------------------------------------------------------------------
# Esquema modificado
# ----------------------------------------------------------------------
INDICADORES_ELIMINADOS = [
    "analfabetismo",
    "trabajo_infantil",
    "sin_aseguramiento",
    "trabajo_informal",
    "agua",
    "excretas",
]

# CAR_trabajo_informal se calcula, pero NO se recomienda como feature: los
# hogares sin PEA quedan privados en desempleo de larga duración (parte del
# objetivo) y casi todos también en trabajo informal, así que funciona como
# un sustituto del objetivo. Para usarla, crea IPM(caracteristicas_nan=True)
# y revisa su redundancia (R0) con desempleo de larga duración.
CARACTERISTICAS_EXCLUIDAS = ["CAR_trabajo_informal"]

# Features recomendadas (columnas CAR_* que devuelve el cortador)
CARACTERISTICAS_NUEVAS = [
    f"CAR_{n}"
    for n in INDICADORES_ELIMINADOS
    if f"CAR_{n}" not in CARACTERISTICAS_EXCLUIDAS
]

# Versión de las características en valores originales (columnas ORG_*).
# - Indicadores de vivienda (sin filtro de personas): el código crudo de cada
#   variable de la ECV, una columna por variable (ORG_P8530, ORG_P8526...).
#   Son categóricas: conviene codificarlas con one-hot.
# - Indicadores de personas (con filtro): proporción de miembros elegibles que
#   cumplen la condición de privación (0 a 1). NaN si el hogar no tiene
#   miembros elegibles. Ej.: ORG_analfabetismo = proporción de personas de 15
#   años o más que no saben leer ni escribir.
ORIGINALES_EXCLUIDAS = ["ORG_trabajo_informal"]  # mismo motivo que CAR_
ORIGINALES_NUEVAS = [
    "ORG_analfabetismo",
    "ORG_trabajo_infantil",
    "ORG_sin_aseguramiento",
    "ORG_P8520S5",  # agua, cabecera: ¿tiene acueducto?
    "ORG_P8530",  # agua, rural: fuente del agua para preparar alimentos
    "ORG_P8520S3",  # excretas, cabecera: ¿tiene alcantarillado?
    "ORG_P8526",  # excretas, rural: tipo de servicio sanitario
]

# Variables crudas que solo usan los indicadores eliminados: se pueden usar
# como features
VARIABLES_LIBERADAS = ["P6160", "P8520S5", "P8530", "P8520S3", "P8526"]

# Variables crudas de indicadores eliminados que también alimentan el corte:
# NO usarlas crudas como features (fuga de información)
VARIABLES_COMPARTIDAS = [
    "P6090",
    "P6240",
    "P6250",
    "P6260",
    "P6270",
    "P6351",
    "P7250",
    "P6920",
    "P8642",
]

# Variables crudas que construyen los 9 indicadores del corte en el esquema
# modificado. Deben excluirse de la matriz de features:
#     X = df.drop(columns=VARIABLES_CORTE)
# P6040 (edad) y CLASE están incluidas porque definen quién es elegible y qué
# umbral aplica; si decides usarlas como features, hazlo de forma explícita.
VARIABLES_CORTE = [
    "P6040",
    "CLASE",
    # logro educativo, rezago e inasistencia escolar
    "P8586",
    "P8587",
    "P8587S1",
    "P6211",
    "P1088",
    "P1088S1",
    "P6216",
    # primera infancia
    "P6090",
    "P51",
    "P55",
    "P774",
    "P6180",
    # barreras de acceso a salud
    "P5665",
    "P8563",
    # desempleo de larga duración (PEA y pensionados)
    "P6240",
    "P6250",
    "P6260",
    "P6270",
    "P6351",
    "P7250",
    "P6920",
    "P8642",
    # vivienda
    "P4015",
    "P4005",
    "P5010",
    "CANT_PERSONAS_HOGAR",
]

# Columnas crudas que lee _preparar
_COLUMNAS_PREPARAR = [
    "P6040",
    "P8586",
    "P8587",
    "P8587S1",
    "P6211",
    "P1088",
    "P1088S1",
    "P6216",
    "P6090",
    "P51",
    "P55",
    "P774",
    "P6180",
    "P6240",
    "P6250",
    "P6260",
    "P6270",
    "P6351",
    "P7250",
    "P6920",
    "P8642",
    "P5010",
    "CANT_PERSONAS_HOGAR",
]

# Columnas que crea _preparar (no se piden en X)
_DERIVADAS = {
    "ANOS_EDUC",
    "BRECHA_EDUC",
    "SIN_CUIDADO_PI",
    "OCUPADO",
    "PEA",
    "PEA2",
    "DLD",
    "FORMAL",
    "SOLO_PENSIONADOS",
    "PERSONAS_CUARTO",
}


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def pesos_equiponderados(indicadores):
    """w del DANE: cada dimensión pesa igual y se reparte entre sus indicadores."""
    dims = pd.Series({n: s["dim"] for n, s in indicadores.items()})
    n_dims = dims.nunique()
    return {n: 1 / n_dims / (dims == d).sum() for n, d in dims.items()}


_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda s, z: s.isin(z),
}


def _comparar(s, op, z, clase=None):
    """Aplica s <op> z; op y z pueden variar según CLASE.

    Siempre convierte s a número, también para "in", para que una columna
    leída como texto ("4") coincida con los códigos numéricos.
    """
    s = pd.to_numeric(s, errors="coerce")
    if not (isinstance(op, dict) or isinstance(z, dict)):
        return _OPS[op](s, z)
    out = pd.Series(False, index=s.index)
    for c in clase.dropna().unique():
        m = clase.eq(c)
        o = op[c] if isinstance(op, dict) else op
        v = z[c] if isinstance(z, dict) else z
        out[m] = _OPS[o](s[m], v)
    return out


def _valor(x, var, clase):
    """Devuelve la variable del indicador; si var es dict, una columna por CLASE."""
    if not isinstance(var, dict):
        return x[var]
    s = pd.Series(np.nan, index=x.index, dtype=float)
    for c, col in var.items():
        m = clase.eq(c)
        s[m] = pd.to_numeric(x.loc[m, col], errors="coerce")
    return s


# ----------------------------------------------------------------------
# Cortador
# ----------------------------------------------------------------------
class IPM(BaseEstimator, TransformerMixin):
    """Índice de Pobreza Multidimensional (Alkire-Foster, doble corte).

    Recibe un DataFrame a nivel persona (una fila por persona) y devuelve
    uno a nivel hogar con las privaciones del corte, las características
    (indicadores fuera del corte, prefijo CAR_), el puntaje c y la marca
    de pobreza.

    Parámetros
    ----------
    indicadores : dict, opcional
        Especificación de indicadores (variables y umbrales z).
        Por defecto DANE_INDICADORES.
    caracteristicas : list, opcional
        Indicadores que se calculan pero no entran al corte (salen como CAR_*).
    w : dict, opcional
        Pesos de los indicadores del corte. Debe cubrir todos los indicadores
        del corte y sumar 1. Por defecto, equiponderación por dimensión.
    k : float, opcional
        Segundo corte. Por defecto 5/15.
    codigos : dict, opcional
        Códigos y umbrales para variables auxiliares. Por defecto DANE_CODIGOS.
    caracteristicas_nan : bool
        Si es True, una característica queda en NaN cuando el hogar no tiene
        miembros elegibles, en lugar de tomar el valor `vacio` del DANE.
    claves_hogar : tuple
        Columnas que identifican al hogar.
    col_peso : str
        Columna con el factor de expansión.
    col_clase : str
        Columna con la zona (1 = cabecera, 2 = resto).
    """

    def __init__(
        self,
        indicadores=None,
        caracteristicas=None,
        w=None,
        k=None,
        codigos=None,
        caracteristicas_nan=False,
        claves_hogar=("DIRECTORIO", "SECUENCIA_P"),
        col_peso="FEX_C",
        col_clase="CLASE",
    ):
        self.indicadores = indicadores
        self.caracteristicas = caracteristicas
        self.w = w
        self.k = k
        self.codigos = codigos
        self.caracteristicas_nan = caracteristicas_nan
        self.claves_hogar = claves_hogar
        self.col_peso = col_peso
        self.col_clase = col_clase

    # ---------------- interfaz sklearn ----------------
    def fit(self, X, y=None):  # noqa: N803 (convención de sklearn)
        self.indicadores_ = self.indicadores or DANE_INDICADORES
        self.caracteristicas_ = list(self.caracteristicas or [])
        self.k_ = DANE_K if self.k is None else self.k
        self.codigos_ = {**DANE_CODIGOS, **(self.codigos or {})}

        desconocidas = set(self.caracteristicas_) - set(self.indicadores_)
        if desconocidas:
            raise ValueError(
                f"Características que no son indicadores: {sorted(desconocidas)}"
            )

        en_corte = {
            n: s for n, s in self.indicadores_.items() if n not in self.caracteristicas_
        }
        self.w_ = dict(self.w) if self.w else pesos_equiponderados(en_corte)

        if set(self.w_) & set(self.caracteristicas_):
            raise ValueError(
                "Un indicador no puede estar en el corte y ser característica: "
                f"{sorted(set(self.w_) & set(self.caracteristicas_))}"
            )
        sobran = set(self.w_) - set(en_corte)
        if sobran:
            raise ValueError(f"Hay pesos sin indicador: {sorted(sobran)}")
        faltan = set(en_corte) - set(self.w_)
        if faltan:
            raise ValueError(
                f"Faltan pesos para indicadores del corte: {sorted(faltan)}"
            )
        if not np.isclose(sum(self.w_.values()), 1):
            raise ValueError(
                f"Los pesos suman {sum(self.w_.values()):.4f}, deben sumar 1"
            )
        return self

    def transform(self, X):  # noqa: N803 (convención de sklearn)
        check_is_fitted(self, "w_")
        self._validar_columnas(X)
        x = self._preparar(X)
        hog = self._privaciones(x)
        hog["C"] = self._puntaje(hog)
        hog["POBRE"] = self._identificar(hog["C"])
        hog = hog.rename(columns={n: f"CAR_{n}" for n in self.caracteristicas_})
        hog = hog.join(self._originales(x))
        return hog.reset_index()[list(self.get_feature_names_out())]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "w_")
        return np.array(
            [
                *self.claves_hogar,
                "PESO",
                *self.w_,
                *[f"CAR_{n}" for n in self.caracteristicas_],
                *self._columnas_org(),
                "C",
                "POBRE",
            ]
        )

    def _columnas_org(self):
        """Nombres de las columnas ORG_* según las características elegidas."""
        cols = []
        for n in self.caracteristicas_:
            s = self.indicadores_[n]
            if s.get("filtro"):
                cols.append(f"ORG_{n}")
            else:
                var = s["var"]
                columnas = var.values() if isinstance(var, dict) else [var]
                cols += [f"ORG_{v}" for v in columnas]
        return list(dict.fromkeys(cols))  # sin duplicados, en orden

    # ---------------- validación de entrada ----------------
    def _columnas_requeridas(self):
        """Columnas crudas que necesita el cálculo con los indicadores actuales."""
        cols = set(_COLUMNAS_PREPARAR)
        cols |= set(self.claves_hogar) | {self.col_peso, self.col_clase}
        for s in self.indicadores_.values():
            var = s["var"]
            cols |= set(var.values()) if isinstance(var, dict) else {var}
            cols |= {col for col, _, _ in s.get("filtro", [])}
        return sorted(cols - _DERIVADAS)

    def _validar_columnas(self, x):
        faltan = [c for c in self._columnas_requeridas() if c not in x.columns]
        if faltan:
            raise ValueError(
                f"Faltan columnas en X: {faltan}. Revisa que los nombres estén en "
                "mayúsculas y que no hayan quedado con sufijos (_x, _y) al "
                "consolidar las tablas."
            )

    # ---------------- fase 0: variables auxiliares ----------------
    def _preparar(self, x):
        """Crea las variables auxiliares a nivel persona que usan los indicadores.

        Todas se calculan fila por fila (persona), salvo SOLO_PENSIONADOS, que
        se calcula por hogar y se repite en cada miembro. Columnas que agrega:
        ANOS_EDUC, BRECHA_EDUC, SIN_CUIDADO_PI, OCUPADO, PEA, DLD, PEA2,
        FORMAL, SOLO_PENSIONADOS y PERSONAS_CUARTO.
        """
        x = x.copy()
        cod = self.codigos_
        claves = [x[c] for c in self.claves_hogar]

        def num(col):
            # Convierte a número; lo que no se pueda convertir queda en NaN
            return pd.to_numeric(x[col], errors="coerce")

        edad = num("P6040")

        # ------------------------------------------------------------------
        # ANOS_EDUC: años de educación formal APROBADOS por la persona.
        # Se cuenta desde primero de primaria (preescolar = 0) y solo se
        # cuentan grados aprobados, así que:
        #   - Si estudia (P8586 == 1), se usa el nivel y grado en que está
        #     matriculado (P1088, P1088S1) y se resta 1, porque el grado en
        #     curso aún no está aprobado.
        #   - Si no estudia, se usa el nivel y último grado aprobado
        #     (P8587, P8587S1).
        #   - En educación superior se suman 11 (bachillerato completo) más
        #     los años de superior aprobados (P6216 si estudia, P6211 si no),
        #     con tope en DANE_CODIGOS["tope_anos_superior"].
        # Supuesto: P8587S1 y P1088S1 registran el grado absoluto (1 a 13),
        # no el grado dentro del nivel.
        # Queda NaN si la persona no tiene información de educación
        # (por ejemplo, menores de 5 años).
        # Lo usa: logro_educativo (promedio del hogar, personas de 15 o más).
        # ------------------------------------------------------------------
        tope = cod["tope_anos_superior"]
        estudia = num("P8586").eq(1)
        na, ga = num("P8587"), num("P8587S1")
        sa = num("P6211").clip(upper=tope)
        nm, gm = num("P1088"), num("P1088S1")
        sm = num("P6216").clip(upper=tope)
        x["ANOS_EDUC"] = np.select(
            [
                estudia & nm.eq(1),  # matriculado en preescolar
                estudia & nm.between(2, 4),  # primaria, secundaria, media: grado - 1
                estudia & nm.between(5, 8),  # superior: 11 + años aprobados
                ~estudia & na.isin([1, 2]),  # ninguno o preescolar
                ~estudia & na.between(3, 5),  # primaria, secundaria, media: grado
                ~estudia & na.between(6, 13),  # superior: 11 + años aprobados
            ],
            [0, (gm - 1).clip(lower=0), 11 + sm.fillna(0), 0, ga, 11 + sa.fillna(0)],
            default=np.nan,
        )

        # ------------------------------------------------------------------
        # BRECHA_EDUC: diferencia entre los años aprobados y los esperados
        # para la edad. Se espera que a los 7 años tenga 1 año aprobado, a
        # los 8 tenga 2, ..., a los 17 tenga 11; es decir, edad - 6.
        #   BRECHA_EDUC = ANOS_EDUC - (P6040 - 6)
        # Negativo = rezago escolar (tiene menos años de los esperados).
        # Lo usa: rezago_escolar (personas de 7 a 17 años, privado si < 0).
        # ------------------------------------------------------------------
        x["BRECHA_EDUC"] = x["ANOS_EDUC"] - (edad - 6)

        # ------------------------------------------------------------------
        # SIN_CUIDADO_PI: 1 si un niño de 0 a 5 años no tiene acceso a
        # servicios de cuidado de la primera infancia, 0 si sí.
        #
        # Para niños de 0 a 4 años (módulo de atención a menores de 5),
        # está privado si cumple al menos una de estas condiciones:
        #   - no está afiliado a salud: P6090 en [2, 9] (no / no sabe);
        #   - pasa la mayor parte del tiempo en un cuidado inadecuado:
        #     P51 en [3, 6, 7] (con padre/madre en el trabajo, con un
        #     pariente menor de 18, o solo en casa);
        #   - asiste a hogar comunitario, jardín, CDI o colegio (P51 == 1)
        #     pero no recibe desayuno o almuerzo gratis o por pago simbólico
        #     (no se cumple P55 == 1 y P774 en [2, 3]).
        #
        # Para niños de 5 años (módulo de educación), está privado si:
        #   - no está afiliado a salud (P6090 en [2, 9]);
        #   - no asiste a una institución educativa (P8586 == 2);
        #   - estudia (P8586 == 1) pero no recibe alimentos en el plantel
        #     gratis o por pago simbólico (P6180 == 2).
        #
        # Variables intermedias:
        #   sin_salud   = P6090 en DANE_CODIGOS["salud_no_afiliado"]
        #   alimento_ok = P55 == 1 y P774 en DANE_CODIGOS["pi_alimento_ok"]
        #   pi_0a4      = condición de privación para 0 a 4 años
        #   pi_5        = condición de privación para 5 años
        # Lo usa: primera_infancia (privado si algún niño de 0 a 5 tiene 1).
        # ------------------------------------------------------------------
        sin_salud = num("P6090").isin(cod["salud_no_afiliado"])
        p51 = num("P51")
        alimento_ok = num("P55").eq(1) & num("P774").isin(cod["pi_alimento_ok"])
        pi_0a4 = edad.le(4) & (
            sin_salud
            | p51.isin(cod["pi_cuidado_inadecuado"])
            | (p51.isin(cod["pi_asiste"]) & ~alimento_ok)
        )
        pi_5 = edad.eq(5) & (
            sin_salud | num("P8586").eq(2) | (num("P8586").eq(1) & num("P6180").eq(2))
        )
        x["SIN_CUIDADO_PI"] = (pi_0a4 | pi_5).astype(int)

        # ------------------------------------------------------------------
        # Mercado laboral. Variables intermedias:
        #   ocupado    = trabajó la semana pasada en cualquiera de estas
        #                formas: P6240 == 1 (su actividad principal fue
        #                trabajar), P6250 == 1 (actividad paga de 1 hora o
        #                más), P6260 == 1 (tenía trabajo o negocio aunque no
        #                trabajó) o P6270 == 1 (trabajo familiar sin pago).
        #   desocupado = no está ocupado y estaba disponible para empezar a
        #                trabajar (P6351 == 1). Sigue el algoritmo del DANE,
        #                que no exige haber buscado trabajo (P6280).
        #   pensionado = ya es pensionado (P6920 == 3) o recibió el mes
        #                pasado ingreso por pensión (P8642 == 1).
        #
        # OCUPADO: 1 si la persona está ocupada (ver `ocupado`).
        #   Lo usa: trabajo_infantil (privado si alguien de 12 a 17 tiene 1).
        #
        # PEA: 1 si pertenece a la población económicamente activa:
        #   tiene 12 años o más (P6040 >= 12) y está ocupada o desocupada.
        #   Lo usa: desempleo_larga, como filtro de personas elegibles.
        #
        # DLD: 1 si es desempleado de larga duración: está desocupado y lleva
        #   buscando trabajo más de DANE_CODIGOS["semanas_dld"] semanas
        #   (P7250 > 52, es decir, más de 12 meses).
        #   Lo usa: desempleo_larga (privado si alguien de la PEA tiene 1).
        #
        # PEA2: 1 si pertenece a la "nueva PEA" del indicador de trabajo
        #   informal: es PEA, pero se excluyen los ocupados menores de 18
        #   años (ya cuentan en trabajo infantil) y los DLD (ya cuentan en
        #   desempleo de larga duración). Incluye a los desocupados de
        #   corta duración.
        #   Lo usa: trabajo_informal, como filtro de personas elegibles.
        #
        # FORMAL: 1 si está ocupado y cotiza a pensión (P6920 == 1) o es
        #   pensionado. Un desocupado de PEA2 tiene FORMAL = 0, así que hace
        #   privado al hogar, como dice el algoritmo del DANE ("la cantidad
        #   de formales es diferente a la cantidad de personas en la nueva
        #   PEA").
        #   Lo usa: trabajo_informal (privado si alguien de PEA2 tiene 0).
        #
        # SOLO_PENSIONADOS: True si TODOS los miembros del hogar son
        #   pensionados (se calcula por hogar y se repite en cada persona).
        #   Un hogar con un niño o con alguien no pensionado queda en False.
        #   Lo usan: desempleo_larga (fuerza 1) y trabajo_informal (fuerza 0).
        # ------------------------------------------------------------------
        ocupado = (
            num("P6240").eq(1)
            | num("P6250").eq(1)
            | num("P6260").eq(1)
            | num("P6270").eq(1)
        )
        desocupado = ~ocupado & num("P6351").eq(1)
        pea = edad.ge(12) & (ocupado | desocupado)
        dld = desocupado & num("P7250").gt(cod["semanas_dld"])
        pensionado = num("P6920").eq(3) | num("P8642").eq(1)

        x["OCUPADO"] = ocupado.astype(int)
        x["PEA"] = pea.astype(int)
        x["DLD"] = dld.astype(int)
        x["PEA2"] = (pea & ~(ocupado & edad.lt(18)) & ~dld).astype(int)
        x["FORMAL"] = (ocupado & (num("P6920").eq(1) | pensionado)).astype(int)
        x["SOLO_PENSIONADOS"] = pensionado.groupby(claves).transform("all")

        # ------------------------------------------------------------------
        # PERSONAS_CUARTO: personas del hogar por cuarto para dormir.
        #   PERSONAS_CUARTO = CANT_PERSONAS_HOGAR / P5010
        # Si P5010 es 0, el resultado queda NaN (se evita dividir por cero)
        # y el hogar NO queda privado en hacinamiento.
        # Lo usa: hacinamiento (privado si >= 3 en cabecera o > 3 en resto).
        # ------------------------------------------------------------------
        cuartos = num("P5010").replace(0, np.nan)
        x["PERSONAS_CUARTO"] = num("CANT_PERSONAS_HOGAR") / cuartos
        return x

    # ---------------- fase 1: primer corte (z) ----------------
    def _privaciones(self, x):
        """Calcula todos los indicadores (corte y características) a nivel hogar."""
        claves = list(self.claves_hogar)
        clase = pd.to_numeric(x[self.col_clase], errors="coerce")
        hog = x.groupby(claves)[self.col_peso].sum().rename("PESO").to_frame()
        solo_pens = (
            x.groupby(claves)["SOLO_PENSIONADOS"]
            .first()
            .reindex(hog.index)
            .fillna(False)
            .astype(bool)
        )

        for nombre, s in self.indicadores_.items():
            m = self._elegibles(x, s)
            xf, cf = x[m], clase[m]
            valor = _valor(xf, s["var"], cf)

            if s.get("agg", "any") == "mean":
                t = (
                    xf[claves]
                    .assign(v=pd.to_numeric(valor, errors="coerce"), c=cf)
                    .groupby(claves)
                    .agg(v=("v", "mean"), c=("c", "first"))
                )
                g = _comparar(t["v"], s["op"], s["z"], t["c"])
            else:
                g = (
                    xf[claves]
                    .assign(g=_comparar(valor, s["op"], s["z"], cf))
                    .groupby(claves)["g"]
                    .max()
                )

            es_car = nombre in self.caracteristicas_
            vacio = (
                np.nan if (self.caracteristicas_nan and es_car) else s.get("vacio", 0)
            )
            g = g.reindex(hog.index).astype(float).fillna(vacio)
            if "solo_pensionados" in s:
                g[solo_pens] = s["solo_pensionados"]
            hog[nombre] = g if g.isna().any() else g.astype(int)
        return hog

    def _elegibles(self, x, s):
        """Máscara True para las personas que cumplen el filtro del indicador."""
        m = pd.Series(True, index=x.index)
        for col, op, z in s.get("filtro", []):
            m &= _comparar(x[col], op, z)
        return m

    def _originales(self, x):
        """Versión de las características en valores originales (columnas ORG_*).

        - Indicador de vivienda (sin filtro): el código crudo de cada variable
          de la ECV, tomado del primer miembro del hogar (es igual en todos).
          Si `var` cambia por zona, sale una columna por cada variable.
        - Indicador de personas con agg="mean": promedio del valor entre los
          miembros elegibles (por ejemplo, años de educación promedio).
        - Indicador de personas con agg="any": proporción de miembros
          elegibles que cumplen la condición de privación (0 a 1).
        Los hogares sin miembros elegibles quedan en NaN.
        """
        claves = list(self.claves_hogar)
        clase = pd.to_numeric(x[self.col_clase], errors="coerce")
        idx = x.groupby(claves).size().index
        org = pd.DataFrame(index=idx)

        for nombre in self.caracteristicas_:
            s = self.indicadores_[nombre]

            if not s.get("filtro"):
                var = s["var"]
                columnas = var.values() if isinstance(var, dict) else [var]
                for col in columnas:
                    org[f"ORG_{col}"] = (
                        pd.to_numeric(x[col], errors="coerce")
                        .groupby([x[c] for c in claves])
                        .first()
                        .reindex(idx)
                    )
                continue

            m = self._elegibles(x, s)
            xf, cf = x[m], clase[m]
            valor = _valor(xf, s["var"], cf)
            if s.get("agg", "any") == "mean":
                v = pd.to_numeric(valor, errors="coerce")
            else:
                v = _comparar(valor, s["op"], s["z"], cf).astype(float)
            org[f"ORG_{nombre}"] = (
                xf[claves].assign(v=v).groupby(claves)["v"].mean().reindex(idx)
            )
        return org

    # ---------------- fase 2: ponderación (w) ----------------
    def _puntaje(self, hog):
        w = pd.Series(self.w_)
        return hog[w.index] @ w

    # ---------------- fase 3: segundo corte (k) ----------------
    def _identificar(self, c):
        return (c >= self.k_ - 1e-9).astype(int)  # tolerancia por redondeo de 5/15

    # ---------------- fase 4: agregación ----------------
    @staticmethod
    def resumen(hog):
        """H, A y M0 a nivel personas (PESO = suma de FEX_C del hogar)."""
        p, pobre, c = hog["PESO"], hog["POBRE"], hog["C"]
        pobres = (p * pobre).sum()
        incidencia = pobres / p.sum()
        intensidad = (p * pobre * c).sum() / pobres if pobres > 0 else np.nan
        m0 = incidencia * intensidad if pobres > 0 else 0.0
        return pd.Series({"H": incidencia, "A": intensidad, "M0": m0})

    # ==================================================================
    # Análisis de robustez del corte (métodos estáticos)
    # ==================================================================
    # Replican tres análisis de Zhao, Cheah y Moses (2024), "Measurement
    # and determinants of multidimensional urban poverty", PLOS ONE 19(5):
    # e0300263, basados en Alkire y Santos (2014):
    #   - Figura 2: robustez al corte k     → dominancia_k + grafico_dominancia
    #   - Tabla 3:  robustez a los indicadores → esquemas_sin_indicador
    #                                           + kendall_esquemas + grafico_kendall
    #   - Tabla 6:  robustez a los pesos     → esquemas_peso_dimension
    #                                           + kendall_esquemas + grafico_kendall
    #
    # Todos reciben la tabla de hogares que devuelve `transform` (necesita
    # PESO y las columnas de privación de los indicadores del corte) y los
    # pesos `w` (por ejemplo, `ipm.w_`). No vuelven a correr `transform`:
    # recalculan C y POBRE con los pesos y el k que se les pase.
    # ==================================================================

    @staticmethod
    def _medidas(hog, w, k):
        """H, A y M0 recalculados con otros pesos `w` y otro corte `k`."""
        w = pd.Series(w, dtype=float)
        c = hog[w.index] @ w
        pobre = (c >= k - 1e-9).astype(int)
        return IPM.resumen(pd.DataFrame({"PESO": hog["PESO"], "POBRE": pobre, "C": c}))

    @staticmethod
    def _grupos(hog, grupo=None, n_grupos=4, semilla=0):
        """Etiqueta de grupo por hogar.

        - `grupo=None`: divide los hogares al azar en `n_grupos` grupos de
          igual tamaño ("Grupo 1", "Grupo 2", ...), como hace el artículo.
        - `grupo="COLUMNA"`: usa esa columna de `hog` (por ejemplo,
          P1_DEPARTAMENTO, si la pegaste a la tabla de hogares).
        - `grupo=serie o arreglo`: usa esos valores, en el orden de `hog`.
        """
        if grupo is None:
            rng = np.random.default_rng(semilla)
            orden = rng.permutation(len(hog))
            etiquetas = np.empty(len(hog), dtype=object)
            for i, parte in enumerate(np.array_split(orden, n_grupos), start=1):
                etiquetas[parte] = f"Grupo {i}"
            return pd.Series(etiquetas, index=hog.index, name="grupo")
        if isinstance(grupo, str):
            return hog[grupo].rename("grupo")
        return pd.Series(np.asarray(grupo), index=hog.index, name="grupo")

    # ---------------- figura 2: robustez al corte k ----------------
    @staticmethod
    def dominancia_k(
        hog, w, ks=(0.2, 0.25, 0.3, 0.4, 0.5), grupo=None, n_grupos=4, semilla=0
    ):
        """M0 de cada grupo para varios valores de k (datos de la figura 2).

        Parámetros
        ----------
        hog : DataFrame
            Salida de `transform`.
        w : dict
            Pesos de los indicadores del corte (por ejemplo, `ipm.w_`).
        ks : valores de k. Por defecto los de la figura 2.
        grupo : None, str o serie. Con None, divide los hogares al azar en
            `n_grupos` grupos de igual tamaño, como el artículo. Ver `_grupos`.
        n_grupos, semilla : solo se usan con `grupo=None`.

        Devuelve
        --------
        DataFrame con una fila por grupo y una columna por k (valores de M0).
        Se grafica con `grafico_dominancia`.

        Cómo interpretar
        ----------------
        Es un análisis de dominancia. El grupo A es "más pobre" que el grupo B
        sin importar k si su M0 es mayor o igual en todos los k probados, y
        estrictamente mayor en al menos uno. En el gráfico, eso se ve como
        dos líneas que no se cruzan.
        - Líneas que no se cruzan: el orden entre esos grupos no depende del
          corte; la conclusión "A es más pobre que B" se sostiene con
          cualquier k del rango.
        - Líneas que se cruzan: el orden depende de k. Si el cruce ocurre
          lejos del k que usas, preocupa poco; si ocurre cerca, la
          comparación entre esos dos grupos no es concluyente.
        - El corte es robusto si la mayoría de los pares de grupos mantienen
          su orden en todo el rango, sobre todo cerca del k elegido. En el
          artículo, el orden de sus cuatro grupos se mantiene en casi todos
          los k, en especial entre 0,25 y 0,3.
        - Todas las líneas bajan al subir k: con un corte más exigente hay
          menos pobres. Eso es esperado y no dice nada sobre la robustez;
          lo que importa es si las líneas se cruzan.
        Nota: con grupos aleatorios, todos vienen de la misma población, así
        que sus diferencias son solo ruido de muestreo y es normal que las
        líneas estén juntas y se crucen. En Colombia es más informativo usar
        grupos reales (departamentos o regiones), porque la ECV sí trae esa
        información y los datos del artículo no.
        """
        g = IPM._grupos(hog, grupo, n_grupos, semilla)
        tabla = pd.DataFrame(
            {
                nombre: {k: IPM._medidas(sub, w, k)["M0"] for k in ks}
                for nombre, sub in hog.groupby(g)
            }
        ).T
        tabla.index.name = "grupo"
        tabla.columns.name = "k"
        return tabla

    # ---------------- tablas 3 y 6: robustez a indicadores y pesos ----------------
    @staticmethod
    def esquemas_sin_indicador(w, indicadores=None, quitar=None):
        """Escenarios de pesos quitando un indicador a la vez (tabla 3).

        Parámetros
        ----------
        w : dict
            Pesos base (por ejemplo, `ipm.w_`).
        indicadores : dict, opcional
            Especificación de indicadores, para saber a qué dimensión
            pertenece cada uno. Por defecto DANE_INDICADORES.
        quitar : lista, opcional
            Indicadores a quitar, uno por escenario. Por defecto, todos los
            de `w`. El artículo quita tres (asistencia escolar, ingreso y
            bienes durables); puedes elegir los que quieras probar.

        Devuelve
        --------
        dict {nombre del escenario: pesos}. Incluye "Base" (los pesos `w`) y
        un escenario "Sin <indicador>" por cada indicador quitado, con los
        pesos recalculados por equiponderación sobre los que quedan. Si el
        indicador era el único de su dimensión, la dimensión desaparece y
        las demás se reparten su peso. Se pasa a `kendall_esquemas`.

        Cómo interpretar
        ----------------
        Ver `kendall_esquemas`. Si quitar un indicador casi no cambia el
        orden de los grupos, la medida no depende de ese indicador para
        decidir qué grupos son más pobres.
        """
        indicadores = indicadores or DANE_INDICADORES
        quitar = list(w) if quitar is None else quitar
        esquemas = {"Base": dict(w)}
        for q in quitar:
            quedan = {n: indicadores[n] for n in w if n != q}
            esquemas[f"Sin {q}"] = pesos_equiponderados(quedan)
        return esquemas

    @staticmethod
    def esquemas_peso_dimension(w, indicadores=None, peso=0.5):
        """Escenarios de pesos dándole más peso a una dimensión (tabla 6).

        Parámetros
        ----------
        w : dict
            Pesos base (por ejemplo, `ipm.w_`). Solo se usan sus nombres.
        indicadores : dict, opcional
            Para saber la dimensión de cada indicador. Por defecto
            DANE_INDICADORES.
        peso : float
            Peso que recibe la dimensión destacada. El resto se reparte por
            igual entre las demás dimensiones, y dentro de cada dimensión
            por igual entre sus indicadores. Con 5 dimensiones y peso=0,5,
            cada una de las otras recibe 0,125 (el artículo tenía 4
            dimensiones, así que las otras recibían 0,1667).

        Devuelve
        --------
        dict {nombre del escenario: pesos}, con "Base" y un escenario
        "<peso>% <dimensión>" por cada dimensión. Se pasa a
        `kendall_esquemas`.

        Cómo interpretar
        ----------------
        Ver `kendall_esquemas`. Si el orden de los grupos se mantiene aunque
        una dimensión pese mucho más que las demás, la equiponderación no
        está determinando qué grupos son más pobres.
        """
        indicadores = indicadores or DANE_INDICADORES
        dims = pd.Series({n: indicadores[n]["dim"] for n in w})
        n_dims = dims.nunique()
        esquemas = {"Base": dict(w)}
        for d in dims.unique():
            peso_dim = {
                x: peso if x == d else (1 - peso) / (n_dims - 1) for x in dims.unique()
            }
            esquemas[f"{peso:.0%} {d}"] = {
                n: peso_dim[dims[n]] / (dims == dims[n]).sum() for n in dims.index
            }
        return esquemas

    @staticmethod
    def kendall_esquemas(hog, esquemas, k=DANE_K, grupo=None, n_grupos=4, semilla=0):
        """Tau de Kendall entre esquemas de indicadores o pesos (tablas 3 y 6).

        Calcula el M0 de cada grupo en cada esquema y compara, para cada par
        de esquemas, el orden de los grupos de más a menos pobre.

        Parámetros
        ----------
        hog : DataFrame
            Salida de `transform`.
        esquemas : dict
            {nombre: pesos}, de `esquemas_sin_indicador`,
            `esquemas_peso_dimension` o armado a mano.
        k : float
            Corte de pobreza. Por defecto 5/15.
        grupo, n_grupos, semilla : ver `dominancia_k`.

        Devuelve
        --------
        (tau, p_valor, m0): tres DataFrames.
        - tau: tau de Kendall entre cada par de esquemas. Solo se llena el
          triángulo inferior, como en el artículo. Se grafica con
          `grafico_kendall`.
        - p_valor: el p-valor de cada tau (prueba de que tau = 0).
        - m0: M0 de cada grupo (filas) en cada esquema (columnas), por si
          quieres ver de dónde salen los ordenamientos.

        Cómo interpretar
        ----------------
        Tau mide cuánto coinciden dos ordenamientos de los grupos: 1 es el
        mismo orden, 0 es ninguna relación y -1 es el orden invertido.
        - Fíjate sobre todo en la primera columna: compara cada escenario
          con la medida base. El artículo toma tau > 0,6 como señal de que
          la medida es robusta: cambiar ese indicador o esos pesos no
          cambia sustancialmente qué grupos son más pobres.
        - Un tau bajo entre "Base" y un escenario dice que esa decisión
          (incluir ese indicador o darle ese peso a la dimensión) reordena
          los grupos: la medida es sensible a ella y conviene justificarla.
        - Las demás celdas comparan escenarios entre sí. Sirven para ver si
          dos cambios mueven los resultados en la misma dirección.
        - Con pocos grupos, tau solo toma unos pocos valores y la prueba casi
          nunca es significativa. Con 4 grupos, tau solo puede valer 1,
          0,67, 0,33, 0, ... y ni siquiera tau = 1 alcanza p < 0,05
          (p = 0,083). El artículo marca significancia con 4 grupos; tómalo
          con cautela. Para un análisis más informativo, usa grupos reales y
          numerosos, como los 33 departamentos (`grupo="P1_DEPARTAMENTO"`).
        """
        g = IPM._grupos(hog, grupo, n_grupos, semilla)
        m0 = pd.DataFrame(
            {
                nombre: {
                    gr: IPM._medidas(sub, w, k)["M0"] for gr, sub in hog.groupby(g)
                }
                for nombre, w in esquemas.items()
            }
        )
        m0.index.name = "grupo"
        nombres = list(esquemas)
        tau = pd.DataFrame(np.nan, index=nombres, columns=nombres)
        p_valor = tau.copy()
        for i, a in enumerate(nombres):
            for b in nombres[: i + 1]:
                t, p = kendalltau(m0[a], m0[b])
                tau.loc[a, b], p_valor.loc[a, b] = t, p
        return tau, p_valor, m0

    # ---------------- gráficos ----------------
    # Paleta categórica en orden fijo (nunca cíclica), tinta neutra para
    # textos y ejes, y formas de marcador distintas por serie para que la
    # identidad no dependa solo del color.
    _COLORES = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ]
    _MARCADORES = ["D", "s", "^", "X", "o", "v", "P", "*"]
    _TINTA = {
        "fondo": "#fcfcfb",
        "texto": "#0b0b0b",
        "secundario": "#52514e",
        "eje": "#c3c2b7",
        "rejilla": "#e1e0d9",
    }

    @staticmethod
    def grafico_dominancia(tabla, titulo=None, k_ref=DANE_K):
        """Gráfico de la figura 2: una línea de M0 por grupo frente a k.

        Parámetros
        ----------
        tabla : DataFrame
            Salida de `dominancia_k`. Máximo 8 grupos, para que cada uno
            tenga un color distinto; con más (por ejemplo, 33
            departamentos), grafica un subconjunto de filas.
        titulo : str, opcional
        k_ref : float o None
            Dibuja una línea vertical en el k que usas. Por defecto 5/15.

        Devuelve
        --------
        La figura de matplotlib (guárdala con `fig.savefig("fig2.png")`).

        Cómo interpretar
        ----------------
        Ver `dominancia_k`: si las líneas no se cruzan, el orden de los
        grupos no depende de k. Fíjate sobre todo en la zona de la línea
        vertical, que marca el k que usas.
        """
        import matplotlib.pyplot as plt

        if len(tabla) > len(IPM._COLORES):
            raise ValueError(
                f"Hay {len(tabla)} grupos y el gráfico admite hasta "
                f"{len(IPM._COLORES)}. Grafica un subconjunto de filas."
            )
        t = IPM._TINTA
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.set_facecolor(t["fondo"])
        ax.set_facecolor(t["fondo"])
        for i, (nombre, fila) in enumerate(tabla.iterrows()):
            ax.plot(
                fila.index.astype(float),
                fila.values,
                color=IPM._COLORES[i],
                marker=IPM._MARCADORES[i],
                markersize=7,
                linewidth=2,
                markeredgecolor=t["fondo"],
                markeredgewidth=1,
                label=str(nombre),
            )
        if k_ref is not None:
            ax.axvline(k_ref, color=t["secundario"], linewidth=1, linestyle="--")
            ax.annotate(
                f"k = {k_ref:.3g}",
                xy=(k_ref, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -2),
                textcoords="offset points",
                va="top",
                color=t["secundario"],
                fontsize=9,
            )
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(t["eje"])
        ax.grid(axis="y", color=t["rejilla"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=t["secundario"], length=0)
        ax.set_xticks(tabla.columns.astype(float))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("k (corte de pobreza)", color=t["secundario"])
        ax.set_ylabel("M0", color=t["secundario"])
        ax.legend(frameon=False, labelcolor=t["texto"])
        if titulo:
            ax.set_title(titulo, loc="left", color=t["texto"], fontsize=12)
        fig.tight_layout()
        return fig

    @staticmethod
    def grafico_kendall(tau, p_valor=None, titulo=None, alfa=0.01):
        """Mapa de calor de la matriz de tau de Kendall (tablas 3 y 6).

        Parámetros
        ----------
        tau : DataFrame
            Primer resultado de `kendall_esquemas`.
        p_valor : DataFrame, opcional
            Segundo resultado de `kendall_esquemas`. Si se pasa, las celdas
            con p < `alfa` llevan "**", como en el artículo.
        titulo : str, opcional
        alfa : float
            Nivel de significancia para el "**". Por defecto 0,01, como el
            artículo.

        Devuelve
        --------
        La figura de matplotlib.

        Cómo interpretar
        ----------------
        Ver `kendall_esquemas`. Solo se muestra el triángulo inferior, porque
        la matriz es simétrica. El color va de naranja (-1, orden invertido)
        a gris (0, sin relación) y a azul (1, mismo orden): una primera
        columna azul intensa quiere decir que ningún escenario cambia mucho
        el orden de los grupos respecto a la medida base.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap

        t = IPM._TINTA
        # Divergente: dos tonos en los extremos y gris neutro en el centro (0)
        cmap = LinearSegmentedColormap.from_list(
            "kendall", ["#eb6834", "#e1e0d9", "#2a78d6"]
        )
        texto = tau.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        if p_valor is not None:
            marca = p_valor.map(lambda p: " **" if pd.notna(p) and p < alfa else "")
            # Sin "**" en la diagonal: cada esquema comparado consigo mismo
            diagonal = np.eye(len(tau), dtype=bool)
            texto = texto + marca.where(tau.notna() & ~diagonal, "")
        mascara = np.triu(np.ones(tau.shape, dtype=bool), k=1)

        n = len(tau)
        fig, ax = plt.subplots(figsize=(2.2 + 1.5 * n, 1.2 + 0.75 * n))
        fig.set_facecolor(t["fondo"])
        sns.heatmap(
            tau.astype(float),
            mask=mascara,
            annot=texto.values,
            fmt="",
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=2,
            linecolor=t["fondo"],
            cbar_kws={"label": "tau de Kendall", "shrink": 0.8},
            annot_kws={"color": t["texto"], "fontsize": 10},
            ax=ax,
        )
        ax.set_facecolor(t["fondo"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors=t["texto"], length=0)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        barra = ax.collections[0].colorbar
        barra.ax.tick_params(colors=t["secundario"], length=0)
        barra.outline.set_visible(False)
        barra.set_label("tau de Kendall", color=t["secundario"])
        if titulo:
            ax.set_title(titulo, loc="left", color=t["texto"], fontsize=12)
        if p_valor is not None:
            fig.text(
                0.01,
                0.01,
                f"** p < {alfa:g} (bilateral)",
                color=t["secundario"],
                fontsize=9,
            )
        fig.tight_layout()
        return fig
