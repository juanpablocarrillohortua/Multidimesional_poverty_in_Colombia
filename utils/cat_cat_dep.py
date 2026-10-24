"""
Hub de pruebas de hipótesis para dependencia entre una variable categórica
base y una lista de variables categóricas, solo con la prueba chi-cuadrado
de independencia.

H0: las dos variables son independientes.

Supuestos que se reportan (no cambian la prueba):
    Tabla de al menos 2x2       -> si no, la fila queda como "no aplica".
    Frecuencias esperadas       -> regla de Cochran:
        2x2: todas las celdas con frecuencia esperada >= 5.
        más grande: ninguna < 1 y como máximo el 20 % de las celdas < 5.
    Independencia de las observaciones -> no se puede verificar con los
        datos; depende del diseño muestral.

Si la tabla es 2x2 (gl = 1), scipy aplica la corrección de Yates. Si se
rechaza H0, el post-hoc usa los residuos estandarizados ajustados de cada
celda con corrección de Bonferroni.

Advertencias:
    - Trabaja con conteos crudos. Una tabla ponderada (por ejemplo con el
      factor de expansión) infla n y da p-valores inválidos, por eso no
      acepta pesos.
    - Con muestreo por conglomerados (SEGMENTO en la ECV) las observaciones
      no son independientes y los p-valores salen optimistas.
    - Con n grande casi todo resulta significativo: lea la V de Cramér
      junto al p-valor.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _supuesto_frecuencias(esperadas):
    """Regla de Cochran. Devuelve (cumple, % celdas < 5, detalle)."""
    e = esperadas.ravel()
    menor_5 = int((e < 5).sum())
    menor_1 = int((e < 1).sum())
    pct_5 = 100 * menor_5 / e.size

    if esperadas.shape == (2, 2):
        cumple = menor_5 == 0
        fallas = (
            []
            if cumple
            else [
                f"{menor_5} de 4 celdas con frecuencia esperada < 5 "
                "(en 2x2 deben ser todas >= 5)"
            ]
        )
    else:
        fallas = []
        if menor_1:
            fallas.append(f"{menor_1} celdas con frecuencia esperada < 1")
        if pct_5 > 20:
            fallas.append(
                f"{menor_5} celdas ({pct_5:.0f} %) con frecuencia "
                "esperada < 5 (máximo 20 %)"
            )
        cumple = not fallas

    if cumple:
        return True, pct_5, "frecuencias esperadas suficientes (Cochran)"
    fallas.append("agrupe las categorías poco frecuentes")
    return False, pct_5, "; ".join(fallas)


def _v_cramer(tabla, esperadas):
    """V de Cramér con el chi-cuadrado sin corrección de Yates."""
    observadas = tabla.to_numpy(dtype=float)
    chi2 = ((observadas - esperadas) ** 2 / esperadas).sum()
    n = observadas.sum()
    return np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))


def _posthoc_residuos(tabla, esperadas, alpha):
    """Celdas con residuo estandarizado ajustado significativo (Bonferroni).

    Signo + : hay más casos de los esperados bajo independencia.
    Signo - : hay menos.
    """
    observadas = tabla.to_numpy(dtype=float)
    n = observadas.sum()
    p_fila = observadas.sum(axis=1, keepdims=True) / n
    p_col = observadas.sum(axis=0, keepdims=True) / n
    residuos = (observadas - esperadas) / np.sqrt(
        esperadas * (1 - p_fila) * (1 - p_col)
    )
    z_critico = stats.norm.ppf(1 - alpha / (2 * observadas.size))

    celdas = [
        (abs(r), f"{fila} × {col} ({'+' if r > 0 else '-'}, r={r:.2f})")
        for i, fila in enumerate(tabla.index)
        for j, col in enumerate(tabla.columns)
        if abs(r := residuos[i, j]) > z_critico
    ]
    celdas.sort(reverse=True)
    if not celdas:
        return "sin celdas significativas"
    return "; ".join(texto for _, texto in celdas)


def hub_pruebas_cat_cat(
    df: pd.DataFrame,
    var_base: str,
    vars_cat: list,
    alpha: float = 0.05,
):
    """
    Parámetros
    ----------
    df       : DataFrame
    var_base : str, variable categórica fija (por ejemplo, el objetivo)
    vars_cat : str o lista de str, variables categóricas a evaluar contra
               var_base
    alpha    : nivel de significancia (default 0.05)

    Retorna
    -------
    DataFrame con una fila por variable de vars_cat, ordenado por p-valor.
    """
    if isinstance(vars_cat, str):
        vars_cat = [vars_cat]

    filas = []
    for cat in vars_cat:
        sub = df[[cat, var_base]].dropna()
        tabla = pd.crosstab(sub[cat], sub[var_base])
        r, c = tabla.shape

        base = {
            "var_base": var_base,
            "var_categorica": cat,
            "tabla": f"{r}x{c}",
            "n": len(sub),
        }

        if r < 2 or c < 2:
            filas.append(
                {
                    **base,
                    "prueba": "no aplica",
                    "estadistico": np.nan,
                    "gl": np.nan,
                    "p_valor": np.nan,
                    "v_cramer": np.nan,
                    "frec_esperada_min": np.nan,
                    "pct_celdas_esp_menor_5": np.nan,
                    "supuesto_frecuencias": np.nan,
                    "supuesto_independencia": np.nan,
                    "detalle_supuestos": "la tabla necesita al menos 2x2",
                    "decision": "no evaluada",
                    "conclusion": "una de las variables tiene un solo nivel",
                    "post_hoc": "",
                }
            )
            continue

        res = stats.chi2_contingency(tabla)
        esperadas = res.expected_freq
        cumple, pct_5, detalle = _supuesto_frecuencias(esperadas)
        rechaza = res.pvalue < alpha

        filas.append(
            {
                **base,
                "prueba": (
                    "chi-cuadrado (Yates)" if res.dof == 1 else "chi-cuadrado"
                ),
                "estadistico": round(float(res.statistic), 4),
                "gl": int(res.dof),
                "p_valor": round(float(res.pvalue), 6),
                "v_cramer": round(float(_v_cramer(tabla, esperadas)), 4),
                "frec_esperada_min": round(float(esperadas.min()), 2),
                "pct_celdas_esp_menor_5": round(pct_5, 1),
                "supuesto_frecuencias": "sí" if cumple else "no",
                "supuesto_independencia": "no verificable (diseño muestral)",
                "detalle_supuestos": detalle,
                "decision": "Se rechaza H0" if rechaza else "No se rechaza H0",
                "conclusion": (
                    "Dependencia: la distribución de una variable cambia "
                    "según la otra"
                    if rechaza
                    else "Independencia: no hay evidencia de asociación"
                ),
                "post_hoc": (
                    _posthoc_residuos(tabla, esperadas, alpha)
                    if rechaza
                    else ""
                ),
            }
        )

    return pd.DataFrame(filas).sort_values("p_valor").reset_index(drop=True)
