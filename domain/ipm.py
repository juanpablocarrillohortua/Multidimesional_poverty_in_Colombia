"""
ipm.py
======

Índice de Pobreza Multidimensional (IPM) para Colombia con el método de
doble corte de Alkire-Foster, siguiendo la metodología del DANE y los
códigos de la Encuesta Nacional de Calidad de Vida (ECV) 2025.

Uso
---
    from ipm import IPM, INDICADORES_ELIMINADOS, CARACTERISTICAS_NUEVAS

    # IPM oficial (15 indicadores)
    hogares = IPM().fit_transform(df)

    # Esquema modificado: 9 indicadores en el corte, 6 como características
    ipm = IPM(caracteristicas=INDICADORES_ELIMINADOS)
    hogares = ipm.fit_transform(df)
    y = hogares["POBRE"]
    X_car = hogares[CARACTERISTICAS_NUEVAS]
    IPM.resumen(hogares)

`df` debe tener una fila por persona, con todas las tablas de la ECV
consolidadas y los nombres de columnas en mayúsculas.
"""

import operator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "IPM",
    "DANE_INDICADORES",
    "DANE_CODIGOS",
    "DANE_K",
    "INDICADORES_ELIMINADOS",
    "CARACTERISTICAS_NUEVAS",
    "VARIABLES_IPM",
    "VARIABLES_LIBERADAS",
    "VARIABLES_COMPARTIDAS",
    "pesos_equiponderados",
]


# ----------------------------------------------------------------------
# Variables de la ECV 2025 usadas en el cálculo
# ----------------------------------------------------------------------
VARIABLES_IPM = {
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
    "logro_educativo": dict(dim="educacion", filtro=[("P6040", ">=", 15)],
                            var="ANOS_EDUC", agg="mean", op="<", z=9, vacio=1),
    "analfabetismo": dict(dim="educacion", filtro=[("P6040", ">=", 15)],
                          var="P6160", op="==", z=2, vacio=1),
    "rezago_escolar": dict(dim="ninez", filtro=[("P6040", ">=", 7), ("P6040", "<=", 17)],
                           var="BRECHA_EDUC", op="<", z=0, vacio=0),
    "inasistencia_escolar": dict(dim="ninez", filtro=[("P6040", ">=", 6), ("P6040", "<=", 16)],
                                 var="P8586", op="==", z=2, vacio=0),
    "primera_infancia": dict(dim="ninez", filtro=[("P6040", "<=", 5)],
                             var="SIN_CUIDADO_PI", op="==", z=1, vacio=0),
    "trabajo_infantil": dict(dim="ninez", filtro=[("P6040", ">=", 12), ("P6040", "<=", 17)],
                             var="OCUPADO", op="==", z=1, vacio=0),
    "sin_aseguramiento": dict(dim="salud", filtro=[("P6040", ">=", 6)],
                              var="P6090", op="in", z=[2, 9], vacio=0),
    "barreras_salud": dict(dim="salud", filtro=[("P5665", "==", 1)],
                           var="P8563", op="in", z=[3, 4, 5, 6, 7, 8], vacio=0),
    "desempleo_larga": dict(dim="trabajo", filtro=[("PEA", "==", 1)],
                            var="P7250", op=">", z=52, vacio=1, solo_pensionados=1),
    "trabajo_informal": dict(dim="trabajo", filtro=[("PEA2", "==", 1)],
                             var="FORMAL", op="==", z=0, vacio=1, solo_pensionados=0),
    "agua": dict(dim="vivienda", var={1: "P8520S5", 2: "P8530"},
                 op={1: "==", 2: "in"}, z={1: 2, 2: [4, 5, 6, 8, 9, 10]}),
    "excretas": dict(dim="vivienda", var={1: "P8520S3", 2: "P8526"},
                     op={1: "==", 2: "in"}, z={1: 2, 2: [3, 5, 6]}),
    "pisos": dict(dim="vivienda", var="P4015", op="==", z=7),
    "paredes": dict(dim="vivienda", var="P4005", op="in",
                    z={1: [5, 7, 8, 9, 10], 2: [7, 8, 9, 10]}),
    "hacinamiento": dict(dim="vivienda", var="PERSONAS_CUARTO",
                         op={1: ">=", 2: ">"}, z=3),
}

# Códigos categóricos usados para construir variables auxiliares
DANE_CODIGOS = {
    "salud_no_afiliado": [2, 9],         # P6090: no / no sabe
    "pi_asiste": [1],                    # P51: hogar comunitario, jardín, CDI o colegio
    "pi_cuidado_inadecuado": [3, 6, 7],  # P51: padre/madre en el trabajo, pariente <18, solo
    "pi_alimento_ok": [2, 3],            # P774: pago simbólico o gratis
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

# Nuevas features (columnas que devuelve el cortador)
CARACTERISTICAS_NUEVAS = [f"CAR_{n}" for n in INDICADORES_ELIMINADOS]

# Variables crudas que solo usan los indicadores eliminados: se pueden usar como features
VARIABLES_LIBERADAS = ["P6160", "P8520S5", "P8530", "P8520S3", "P8526"]

# Variables crudas de indicadores eliminados que también alimentan el corte:
# NO usarlas crudas como features (fuga de información)
VARIABLES_COMPARTIDAS = ["P6090", "P6240", "P6250", "P6260", "P6270", "P6920", "P8642"]


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def pesos_equiponderados(indicadores):
    """w del DANE: cada dimensión pesa igual y se reparte igual entre sus indicadores."""
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
    """Aplica s <op> z; op y z pueden variar según CLASE."""
    s = s if op == "in" else pd.to_numeric(s, errors="coerce")
    if not (isinstance(op, dict) or isinstance(z, dict)):
        return _OPS[op](s, z)
    out = pd.Series(False, index=s.index)
    for c in clase.dropna().unique():
        m = clase.eq(c)
        o = op[c] if isinstance(op, dict) else op
        v = z[c] if isinstance(z, dict) else z
        out[m] = _OPS[o](s[m], v)
    return out


def _valor(X, var, clase):
    """Devuelve la variable del indicador; si var es dict, toma una columna por CLASE."""
    if not isinstance(var, dict):
        return X[var]
    s = pd.Series(np.nan, index=X.index, dtype=float)
    for c, col in var.items():
        m = clase.eq(c)
        s[m] = pd.to_numeric(X.loc[m, col], errors="coerce")
    return s


# ----------------------------------------------------------------------
# Cortador
# ----------------------------------------------------------------------
class IPM(BaseEstimator, TransformerMixin):
    """Índice de Pobreza Multidimensional (Alkire-Foster, doble corte).

    Recibe un DataFrame a nivel persona (una fila por persona) y devuelve
    uno a nivel hogar con las privaciones del corte, las características
    (indicadores fuera del corte, prefijo CAR_), el puntaje c y la marca de pobreza.

    Parámetros
    ----------
    indicadores : dict, opcional
        Especificación de indicadores (variables y umbrales z). Por defecto DANE_INDICADORES.
    caracteristicas : list, opcional
        Indicadores que se calculan pero no entran al corte (salen como CAR_*).
    w : dict, opcional
        Pesos de los indicadores del corte. Por defecto, equiponderación por dimensión.
    k : float, opcional
        Segundo corte. Por defecto 5/15.
    codigos : dict, opcional
        Códigos categóricos para variables auxiliares. Por defecto DANE_CODIGOS.
    claves_hogar : tuple
        Columnas que identifican al hogar.
    col_peso : str
        Columna con el factor de expansión.
    col_clase : str
        Columna con la zona (1 = cabecera, 2 = resto).
    """

    def __init__(self, indicadores=None, caracteristicas=None, w=None, k=None,
                 codigos=None, claves_hogar=("DIRECTORIO", "SECUENCIA_P"),
                 col_peso="FEX_C", col_clase="CLASE"):
        self.indicadores = indicadores
        self.caracteristicas = caracteristicas
        self.w = w
        self.k = k
        self.codigos = codigos
        self.claves_hogar = claves_hogar
        self.col_peso = col_peso
        self.col_clase = col_clase

    # ---------------- interfaz sklearn ----------------
    def fit(self, X, y=None):
        self.indicadores_ = self.indicadores or DANE_INDICADORES
        self.caracteristicas_ = list(self.caracteristicas or [])
        self.k_ = DANE_K if self.k is None else self.k
        self.codigos_ = self.codigos or DANE_CODIGOS

        desconocidas = set(self.caracteristicas_) - set(self.indicadores_)
        if desconocidas:
            raise ValueError(f"Características que no son indicadores: {desconocidas}")

        en_corte = {n: s for n, s in self.indicadores_.items()
                    if n not in self.caracteristicas_}
        self.w_ = self.w or pesos_equiponderados(en_corte)

        if set(self.w_) & set(self.caracteristicas_):
            raise ValueError("Un indicador no puede estar en el corte y ser característica")
        if set(self.w_) - set(en_corte):
            raise ValueError(f"Hay pesos sin indicador: {set(self.w_) - set(en_corte)}")
        if not np.isclose(sum(self.w_.values()), 1):
            raise ValueError(f"Los pesos suman {sum(self.w_.values()):.4f}, deben sumar 1")
        return self

    def transform(self, X):
        X = self._preparar(X)
        hog = self._privaciones(X)
        hog["C"] = self._puntaje(hog)
        hog["POBRE"] = self._identificar(hog["C"])
        hog = hog.rename(columns={n: f"CAR_{n}" for n in self.caracteristicas_})
        return hog.reset_index()[list(self.get_feature_names_out())]

    def get_feature_names_out(self, input_features=None):
        return np.array([*self.claves_hogar, "PESO", *self.w_,
                         *[f"CAR_{n}" for n in self.caracteristicas_], "C", "POBRE"])

    # ---------------- fase 0: variables auxiliares ----------------
    def _preparar(self, X):
        X = X.copy()
        cod = self.codigos_
        claves = [X[c] for c in self.claves_hogar]
        num = lambda c: pd.to_numeric(X[c], errors="coerce")
        edad = num("P6040")

        # Años de educación aprobados
        estudia = num("P8586").eq(1)
        na, ga, sa = num("P8587"), num("P8587S1"), num("P6211")
        nm, gm, sm = num("P1088"), num("P1088S1"), num("P6216")
        X["ANOS_EDUC"] = np.select(
            [estudia & nm.eq(1),               # matriculado en preescolar
             estudia & nm.between(2, 4),       # primaria, secundaria, media: grado - 1
             estudia & nm.between(5, 8),       # superior: 11 + años aprobados
             ~estudia & na.isin([1, 2]),       # ninguno o preescolar
             ~estudia & na.between(3, 5),      # primaria, secundaria, media: grado
             ~estudia & na.between(6, 13)],    # superior: 11 + años aprobados
            [0, (gm - 1).clip(lower=0), 11 + sm.fillna(0),
             0, ga, 11 + sa.fillna(0)],
            default=np.nan)
        X["BRECHA_EDUC"] = X["ANOS_EDUC"] - (edad - 6)

        # Primera infancia
        sin_salud = num("P6090").isin(cod["salud_no_afiliado"])
        p51 = num("P51")
        alimento_ok = num("P55").eq(1) & num("P774").isin(cod["pi_alimento_ok"])
        pi_0a4 = edad.le(4) & (sin_salud
                               | p51.isin(cod["pi_cuidado_inadecuado"])
                               | (p51.isin(cod["pi_asiste"]) & ~alimento_ok))
        pi_5 = edad.eq(5) & (sin_salud
                             | num("P8586").eq(2)
                             | (num("P8586").eq(1) & num("P6180").eq(2)))
        X["SIN_CUIDADO_PI"] = (pi_0a4 | pi_5).astype(int)

        # Mercado laboral
        z_dld = self.indicadores_.get("desempleo_larga", {}).get("z", 52)
        ocupado = (num("P6240").eq(1) | num("P6250").eq(1)
                   | num("P6260").eq(1) | num("P6270").eq(1))
        desocupado = ~ocupado & num("P6351").eq(1)
        pea = edad.ge(12) & (ocupado | desocupado)
        dld = desocupado & num("P7250").gt(z_dld)
        pensionado = num("P6920").eq(3) | num("P8642").eq(1)

        X["OCUPADO"] = ocupado.astype(int)
        X["PEA"] = pea.astype(int)
        X["PEA2"] = (pea & ~(ocupado & edad.lt(18)) & ~dld).astype(int)
        X["FORMAL"] = (ocupado & (num("P6920").eq(1) | pensionado)).astype(int)
        X["SOLO_PENSIONADOS"] = pensionado.groupby(claves).transform("all")

        # Hacinamiento
        X["PERSONAS_CUARTO"] = num("CANT_PERSONAS_HOGAR") / num("P5010").replace(0, np.nan)
        return X

    # ---------------- fase 1: primer corte (z) ----------------
    def _privaciones(self, X):
        """Calcula todos los indicadores (corte y características) a nivel hogar."""
        claves = list(self.claves_hogar)
        clase = pd.to_numeric(X[self.col_clase], errors="coerce")
        hog = X.groupby(claves)[self.col_peso].sum().rename("PESO").to_frame()
        solo_pens = X.groupby(claves)["SOLO_PENSIONADOS"].first().reindex(hog.index)

        for nombre, s in self.indicadores_.items():
            m = pd.Series(True, index=X.index)
            for col, op, z in s.get("filtro", []):
                m &= _comparar(X[col], op, z)
            Xf, cf = X[m], clase[m]
            valor = _valor(Xf, s["var"], cf)

            if s.get("agg", "any") == "mean":
                t = (Xf[claves].assign(v=pd.to_numeric(valor, errors="coerce"), c=cf)
                     .groupby(claves).agg(v=("v", "mean"), c=("c", "first")))
                g = _comparar(t["v"], s["op"], s["z"], t["c"])
            else:
                g = (Xf[claves].assign(g=_comparar(valor, s["op"], s["z"], cf))
                     .groupby(claves)["g"].max())

            g = g.reindex(hog.index).fillna(s.get("vacio", 0)).astype(int)
            if "solo_pensionados" in s:
                g[solo_pens.fillna(False).astype(bool)] = s["solo_pensionados"]
            hog[nombre] = g
        return hog

    # ---------------- fase 2: ponderación (w) ----------------
    def _puntaje(self, hog):
        w = pd.Series(self.w_)
        return hog[w.index] @ w

    # ---------------- fase 3: segundo corte (k) ----------------
    def _identificar(self, c):
        return (c >= self.k_ - 1e-9).astype(int)   # tolerancia por redondeo de 5/15

    # ---------------- fase 4: agregación ----------------
    @staticmethod
    def resumen(hog):
        """H, A y M0 a nivel personas (PESO = suma de FEX_C del hogar)."""
        p, pobre, c = hog["PESO"], hog["POBRE"], hog["C"]
        H = (p * pobre).sum() / p.sum()
        A = (p * pobre * c).sum() / (p * pobre).sum()
        return pd.Series({"H": H, "A": A, "M0": H * A})
