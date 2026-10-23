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
    X_car = hogares[CARACTERISTICAS_NUEVAS]
    IPM.resumen(hogares)

`df` debe tener una fila por persona, con todas las tablas de la ECV
consolidadas y los nombres de columnas en mayúsculas.

Nota sobre sklearn: IPM convierte filas de personas en filas de hogares,
así que no puede ir antes de un modelo dentro de un mismo Pipeline. Córrelo
una vez antes de partir los datos (fit no aprende nada de X, así que no hay
fuga) y luego divide por hogar o segmento, por ejemplo con GroupKFold.
"""

import operator

import numpy as np
import pandas as pd
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
        Para usar el metodo mas cercano al del DANE dejelo vacio.
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
        return hog.reset_index()[list(self.get_feature_names_out())]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "w_")
        return np.array(
            [
                *self.claves_hogar,
                "PESO",
                *self.w_,
                *[f"CAR_{n}" for n in self.caracteristicas_],
                "C",
                "POBRE",
            ]
        )

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
        na, ga = num("P8587"), num("P8587S1") # nivel alcanzado, grado alcanzado (ultimo aprovado)
        sa = num("P6211").clip(upper=tope) # Años de superior alcanzado (no semestres)
        nm, gm = num("P1088"), num("P1088S1") # nivel matriculado, grado matriculado
        sm = num("P6216").clip(upper=tope) # Años de superior aprovados
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
        """
        Calcula todos los indicadores (corte y características) a nivel hogar.

        Cada indicador es binario, esta privado si o no.
        """
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
            m = pd.Series(True, index=x.index)
            for col, op, z in s.get("filtro", []):
                m &= _comparar(x[col], op, z)
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

    # ---------------- fase 2: ponderación (w) ----------------
    def _puntaje(self, hog):
        """Retorna el puntaje de privacion multidimensional de los hogares"""
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
        incidencia = (p * pobre).sum() / p.sum()
        intensidad = (p * pobre * c).sum() / (p * pobre).sum()
        return pd.Series(
            {"H": incidencia, "A": intensidad, "M0": incidencia * intensidad}
        )
