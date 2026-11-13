"""
Evaluación de un Pipeline de clasificación binaria con validación cruzada
agrupada y estratificada, con un umbral de decisión explícito.

Reúne en una clase el flujo de `notebooks/modeling.ipynb`: validación
cruzada, resumen de métricas, frecuencia de selección de variables, curvas
precisión-recall, compromiso precisión-recall según el umbral, comparación
de métricas en train y test, matriz de confusión, intervalos de confianza
por bootstrap y ajuste final sobre todo train.

Uso::

    ev = EvaluadorCV(pipeline, X_train, y_train, 0.5, grupos_train)
    ev.evaluar_cv().resumen_metricas()
    ev.curvas_pr()
    modelo = ev.ajustar_final()

Los grupos (por ejemplo, SEGMENTO) llegan también al CV interno de pasos
como SequentialFeatureSelector gracias al enrutamiento de metadatos de
scikit-learn, que se activa solo durante cada llamada y no cambia la
configuración global.
"""

import inspect
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    get_scorer,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_validate,
)

from utils.cache import cached

# data/ está en .gitignore, así que la caché nunca se versiona
DIR_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache_cv"


def _custom_accuracy(y_true, y_proba, threshold):
    return accuracy_score(y_true, y_proba >= threshold)


def _custom_precision(y_true, y_proba, threshold):
    return precision_score(y_true, y_proba >= threshold, zero_division=0)


def _custom_recall(y_true, y_proba, threshold):
    return recall_score(y_true, y_proba >= threshold, zero_division=0)


def _custom_kappa(y_true, y_proba, threshold):
    return cohen_kappa_score(y_true, y_proba >= threshold)


def _crear_scoring(umbral):
    """Scorers con el umbral explícito."""

    def con_umbral(funcion):
        return make_scorer(
            funcion, response_method="predict_proba", threshold=umbral
        )

    return {
        "accuracy": con_umbral(_custom_accuracy),
        "precision": con_umbral(_custom_precision),
        "recall": con_umbral(_custom_recall),
        "kappa": con_umbral(_custom_kappa),
        # No depende del umbral: AP evalúa todos los umbrales
        "average_precision": "average_precision",
    }


def _fuente_scorers():
    """Código de los scorers: entra en la clave de la caché para que
    cambiarlos invalide los resultados guardados."""
    return "".join(
        inspect.getsource(f)
        for f in (
            _custom_accuracy,
            _custom_precision,
            _custom_recall,
            _custom_kappa,
            _crear_scoring,
        )
    )


def _validar_cruzado(
    estimador, X, y, grupos, umbral, cv, version_sklearn, fuente_scorers
):
    """Validación cruzada y probabilidades fuera de muestra por pliegue.

    Es la parte lenta de EvaluadorCV.evaluar_cv() y la que se guarda en
    caché. version_sklearn y fuente_scorers no se usan aquí: solo entran
    en la clave de la caché.
    """
    with sklearn.config_context(enable_metadata_routing=True):
        scores = cross_validate(
            estimator=estimador,
            X=X,
            y=y,
            params={"groups": grupos},
            cv=cv,
            scoring=_crear_scoring(umbral),
            return_estimator=True,
            return_indices=True,
        )

    # Probabilidades de cada pliegue con los modelos que ya ajustó
    # cross_validate: no se reentrena nada
    probas_cv = [
        (y.iloc[idx_te], modelo.predict_proba(X.iloc[idx_te])[:, 1])
        for modelo, idx_te in zip(
            scores["estimator"], scores["indices"]["test"]
        )
    ]
    return {"scores": scores, "probas_cv": probas_cv}


def _cox_snell_r2(model, X, y):
    """
    Calcula el R² de Cox & Snell para Regresión Logística.
    Requiere probabilidades predichas del modelo.
    """
    n = len(y)
    # Log-likelihood del modelo actual
    probs = model.predict_proba(X)
    ll_model = -log_loss(y, probs, normalize=False)

    # Log-likelihood del modelo nulo (solo intercepto)
    p_null = np.mean(y)
    ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))

    return 1 - np.exp((2 / n) * (ll_null - ll_model))


def _nagelkerke_r2(model, X, y):
    """
    Calcula el R² de Nagelkerke ajustando la escala entre 0 y 1.
    """
    n = len(y)
    p_null = np.mean(y)
    ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))

    cs_r2 = _cox_snell_r2(model, X, y)
    max_r2 = 1 - np.exp((2 / n) * ll_null)

    return cs_r2 / max_r2


class EvaluadorCV:
    """Validación cruzada y diagnóstico de un Pipeline de clasificación.

    Parámetros
    ----------
    pipeline     : Pipeline (o estimador) con predict_proba
    X_train      : DataFrame de características
    y_train      : Series con el objetivo binario (0/1)
    umbral       : umbral de decisión para accuracy, precision, recall,
                   kappa y la matriz de confusión
    grupos       : Series con el grupo de cada fila (por ejemplo, SEGMENTO)
    cv           : separador de pliegues. Por defecto
                   StratifiedGroupKFold(5, shuffle=True, random_state)
    grid         : param_grid para GridSearchCV (por ejemplo
                   {"model__C": [0.1, 1, 10]}). Si se da, cada pliegue
                   busca los hiperparámetros solo con sus filas de
                   entrenamiento (validación cruzada anidada), con un
                   StratifiedGroupKFold(5) interno de semilla
                   random_state + 1. Por defecto None: sin búsqueda
    target_score : métrica con la que GridSearchCV elige los
                   hiperparámetros (default "average_precision")
    n_jobs       : procesos de GridSearchCV (default -1, todos)
    random_state : semilla del separador por defecto (default 42)
    cache        : si es True (default), evaluar_cv() guarda sus
                   resultados con joblib (utils/cache.py) y, en las
                   siguientes ejecuciones, los carga en vez de reentrenar
                   mientras no cambie nada de lo que los determina:
                   Pipeline (y grid), datos, grupos, umbral, pliegues,
                   versión de scikit-learn, el código de la validación
                   cruzada o el de los scorers. Los datos se comparan por
                   contenido, no por cómo están guardados en memoria.
    pickle       : alias antiguo de cache. Si se da, manda sobre cache
    dir_cache    : carpeta de los archivos. Por defecto data/cache_cv/ en
                   la raíz del repositorio. Cargue solo archivos propios:
                   un archivo de joblib puede ejecutar código al abrirse.
    """

    def __init__(
        self,
        pipeline,
        X_train,
        y_train,
        umbral,
        grupos,
        cv=None,
        grid=None,
        target_score="average_precision",
        n_jobs=-1,
        random_state=42,
        cache=True,
        pickle=None,
        dir_cache=None,
    ):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.umbral = umbral
        self.grupos = grupos
        if cv is None:
            cv = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=random_state
            )
        # Pliegues internos de GridSearchCV, distintos de los externos
        self.cv_grid = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=random_state + 1
        )
        self.cv = cv
        self.random_state = random_state
        self.cache = cache if pickle is None else pickle
        self.dir_cache = Path(dir_cache) if dir_cache else DIR_CACHE
        self.scores = None
        self.probas_cv = None
        self.k_intersect = None
        self.train_test = None
        self.bootstrap = None
        self.grid = grid
        self.target_score = target_score
        self.n_jobs = n_jobs

    def _requiere_cv(self):
        if self.scores is None:
            raise ValueError("Primero ejecute evaluar_cv().")

    def _estimador(self):
        """Pipeline a validar; envuelto en GridSearchCV si hay grid.

        No modifica self.pipeline, así que se puede llamar varias veces.
        """
        if self.grid is None:
            return self.pipeline
        return GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.grid,
            cv=self.cv_grid,
            scoring=self.target_score,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _ajustado(estimador):
        """Pipeline de un pliegue: el mejor de la búsqueda si hubo grid."""
        return getattr(estimador, "best_estimator_", estimador)

    def mejores_parametros(self, estabilidad=False):
        """Hiperparámetros que GridSearchCV eligió en cada pliegue.

        Con estabilidad=False (default) devuelve un DataFrame con una fila
        por pliegue externo y una columna por hiperparámetro.

        Con estabilidad=True imprime un resumen y devuelve un dict con:

        por_pliegue : la misma tabla de una fila por pliegue
        conjuntos   : cada combinación completa distinta, con cuántos
                      pliegues la eligieron (Pliegues, Porcentaje) y cuáles
                      (Folds). La estabilidad del conjunto es el Porcentaje
                      de la primera fila: 100 % si todos los pliegues
                      eligieron la misma combinación
        valores     : cada valor de cada hiperparámetro, con cuántos
                      pliegues lo eligieron (Pliegues, Porcentaje)
        parametros  : por hiperparámetro, el valor más elegido
                      (Valor_modal), en cuántos pliegues (Pliegues_modal),
                      su porcentaje (Estabilidad) y cuántos valores
                      distintos salieron (Valores_distintos)

        En las tres tablas de estabilidad, los valores que no son números
        ni texto (None o, por ejemplo, un escalador dentro del grid) se
        comparan por su repr. Si un hiperparámetro no aplica en un pliegue
        (grid como lista de dicts), queda como NaN y cuenta como un valor
        más ("no aplica" en el resumen).
        """
        self._requiere_cv()
        if self.grid is None:
            raise ValueError(
                "No hubo búsqueda de hiperparámetros: pase grid al crear "
                "el EvaluadorCV."
            )
        por_pliegue = pd.DataFrame(
            [estimador.best_params_ for estimador in self.scores["estimator"]]
        ).rename_axis("fold")
        if not estabilidad:
            return por_pliegue

        def etiqueta(valor):
            # None queda como texto para no confundirlo con un
            # hiperparámetro que no aplica (NaN); los objetos
            # (estimadores) no se comparan bien entre clones
            if isinstance(valor, (bool, int, float, str)):
                return valor
            return repr(valor)

        def texto(valor):
            return "no aplica" if pd.isna(valor) else str(valor)

        params = list(por_pliegue.columns)
        # dtype object: los enteros no pasan a float por un NaN
        etiquetas = pd.DataFrame(
            [
                {k: etiqueta(v) for k, v in estimador.best_params_.items()}
                for estimador in self.scores["estimator"]
            ],
            columns=params,
            dtype=object,
        ).rename_axis("fold")
        n_folds = len(etiquetas)

        # Combinación completa elegida en cada pliegue
        conjuntos = (
            etiquetas.reset_index()
            .groupby(params, dropna=False, sort=False)["fold"]
            .agg(Pliegues="count", Folds=list)
            .reset_index()
            .sort_values("Pliegues", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        conjuntos["Porcentaje"] = conjuntos["Pliegues"] / n_folds * 100

        # Cada valor de cada hiperparámetro por separado
        valores = (
            etiquetas.melt(var_name="Parametro", value_name="Valor")
            .groupby(["Parametro", "Valor"], dropna=False, sort=False)
            .size()
            .rename("Pliegues")
            .reset_index()
            .sort_values(
                ["Parametro", "Pliegues"],
                ascending=[True, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        valores["Porcentaje"] = valores["Pliegues"] / n_folds * 100

        # Valor modal de cada hiperparámetro (en empate, el primero que
        # apareció)
        modal = (
            valores.groupby("Parametro", sort=False)
            .head(1)
            .set_index("Parametro")
        )
        parametros = pd.DataFrame(
            {
                "Valor_modal": modal["Valor"],
                "Pliegues_modal": modal["Pliegues"],
                "Estabilidad": modal["Porcentaje"],
                "Valores_distintos": valores.groupby("Parametro").size(),
            }
        ).reindex(params)
        parametros.index.name = "Parametro"

        primero = conjuntos.iloc[0]
        combinacion = ", ".join(f"{p}={texto(primero[p])}" for p in params)
        print(
            f"Conjunto más elegido: {combinacion} en "
            f"{int(primero['Pliegues'])}/{n_folds} pliegues "
            f"({primero['Porcentaje']:.0f} %); "
            f"{len(conjuntos)} combinaciones distintas."
        )
        for param in params:
            print(
                f"  {param}: {texto(parametros.at[param, 'Valor_modal'])} "
                f"en {int(parametros.at[param, 'Pliegues_modal'])}/{n_folds}"
                f" pliegues ({parametros.at[param, 'Estabilidad']:.0f} %), "
                f"{int(parametros.at[param, 'Valores_distintos'])} valores "
                "distintos"
            )

        return {
            "por_pliegue": por_pliegue,
            "conjuntos": conjuntos,
            "valores": valores,
            "parametros": parametros,
        }

    def evaluar_cv(self, recalcular=False):
        """Corre la validación cruzada y guarda las probabilidades por
        pliegue. Devuelve self para poder encadenar llamadas.

        Con la caché activa (por defecto) carga los resultados si ya
        existen para estos mismos parámetros, sin reentrenar. Con
        recalcular=True reentrena y sobrescribe el archivo guardado.
        """
        print(f"umbral: {self.umbral}")
        validar = _validar_cruzado
        opciones = {}
        if self.cache:
            validar = cached(cache_dir=self.dir_cache, label="cv")(validar)
            opciones = {"force_rerun": recalcular}

        # clone() deja solo los hiperparámetros, así que un Pipeline ya
        # ajustado da la misma clave de caché
        resultado = validar(
            clone(self._estimador()),
            self.X_train,
            self.y_train,
            self.grupos,
            self.umbral,
            self.cv,
            sklearn.__version__,
            _fuente_scorers(),
            **opciones,
        )
        self.scores = resultado["scores"]
        self.probas_cv = resultado["probas_cv"]
        return self

    def resumen_metricas(self):
        """Media ± desviación estándar de cada métrica entre pliegues."""
        self._requiere_cv()
        for metric, val in self.scores.items():
            # 'estimator' e 'indices' no son métricas
            if metric not in ("estimator", "indices"):
                print(f"{metric}: {np.mean(val):.3f} ± {np.std(val):.3f}")

    def frecuencia_variables(self):
        """Cuántas veces se seleccionó cada variable entre los pliegues.

        Requiere pasos llamados 'imputer' y 'select_vars' en el Pipeline.
        """
        self._requiere_cv()
        pasos = getattr(self.pipeline, "named_steps", {})
        if not {"imputer", "select_vars"} <= set(pasos):
            raise ValueError(
                "El Pipeline necesita pasos llamados 'imputer' y "
                "'select_vars' para contar la selección de variables."
            )

        feature_counts = Counter()

        for estimador in self.scores["estimator"]:
            estimator = self._ajustado(estimador)
            # Nombres de variables después del imputer
            feat_names = estimator.named_steps[
                "imputer"
            ].get_feature_names_out(self.X_train.columns)

            # Máscara de las variables seleccionadas en este fold
            mask = estimator.named_steps["select_vars"].get_support()

            # Acumular las variables seleccionadas
            selected_in_fold = feat_names[mask]
            feature_counts.update(selected_in_fold)

        # 2. Convertir a DataFrame para presentar los resultados organizados
        n_folds = len(self.scores["estimator"])
        df_frequencies = (
            pd.DataFrame(
                feature_counts.items(), columns=["Variable", "Frecuencia"]
            )
            .sort_values(by="Frecuencia", ascending=False)
            .reset_index(drop=True)
        )

        # Agregar la proporción/porcentaje de selección
        df_frequencies["Porcentaje"] = (
            df_frequencies["Frecuencia"] / n_folds
        ) * 100

        print(
            f"--- Frecuencia de selección de variables ({n_folds} Folds) ---"
        )
        print(df_frequencies.to_string(index=False))
        return df_frequencies

    def coeficientes(self):
        """Coeficientes del modelo por variable seleccionada, entre pliegues.

        Requiere pasos llamados 'imputer', 'select_vars' y 'model', y un
        modelo con coef_ (lineal).

        Si el modelo es una regresión logística, agrega el AOR (odds ratio
        ajustado): exp del coeficiente promedio, es decir, la media
        geométrica de los odds ratio de los pliegues, y su rango entre
        pliegues (AOR_Min, AOR_Max). Un AOR > 1 aumenta las odds del
        objetivo y uno < 1 las reduce, por cada unidad de la variable
        (de 0 a 1 en las proporciones, una posición en los códigos
        ordinales) con las demás variables seleccionadas fijas.

        Cautelas: LogisticRegression penaliza por defecto (L2, C=1), lo que
        acerca un poco el AOR a 1; class_weight="balanced" mueve sobre todo
        el intercepto, así que las pendientes siguen siendo interpretables.
        """
        self._requiere_cv()
        pasos = getattr(self.pipeline, "named_steps", {})
        if not {"imputer", "select_vars", "model"} <= set(pasos):
            raise ValueError(
                "El Pipeline necesita pasos llamados 'imputer', "
                "'select_vars' y 'model' para resumir los coeficientes."
            )
        primero = self._ajustado(self.scores["estimator"][0])
        if not hasattr(primero["model"], "coef_"):
            raise ValueError(
                "El modelo no tiene coeficientes (coef_): solo aplica a "
                "modelos lineales."
            )

        registros = []
        n_folds = len(self.scores["estimator"])

        for idx, estimador in enumerate(self.scores["estimator"]):
            pipeline = self._ajustado(estimador)
            imputer = pipeline.named_steps["imputer"]
            selector = pipeline.named_steps["select_vars"]
            modelo = pipeline.named_steps["model"]

            # 1. Recuperar los nombres de las columnas que genera el
            # SimpleImputer (incluye las variables originales y las
            # columnas indicadoras si hubo nulos)
            nombres_imputed = imputer.get_feature_names_out(
                self.X_train.columns
            )

            # 2. Filtrar las columnas seleccionadas por el
            # SequentialFeatureSelector
            nombres_seleccionados = nombres_imputed[selector.get_support()]

            # 3. Guardar variables y coeficientes de este fold
            for var, coef in zip(nombres_seleccionados, modelo.coef_[0]):
                registros.append(
                    {"fold": idx, "Característica": var, "Coeficiente": coef}
                )

        df_coefs_todos = pd.DataFrame(registros)

        # 4. Agrupar para ver promedio, desviación estándar y en cuántos
        # folds fue seleccionada cada variable
        df_resumen_coef = (
            df_coefs_todos.groupby("Característica")["Coeficiente"]
            .agg(
                Coef_Promedio="mean",
                Coef_Std="std",
                Veces_Seleccionada="count",
            )
            .sort_values(by="Veces_Seleccionada", ascending=False)
        )

        df_resumen_coef["Porcentaje"] = (
            df_resumen_coef["Veces_Seleccionada"] / n_folds
        ) * 100

        # 5. Regresión logística: odds ratio ajustado (escala log-odds
        # promediada y luego exponenciada)
        if isinstance(pasos["model"], LogisticRegression):
            coefs = df_coefs_todos.groupby("Característica")["Coeficiente"]
            df_resumen_coef["AOR"] = np.exp(df_resumen_coef["Coef_Promedio"])
            df_resumen_coef["AOR_Min"] = np.exp(coefs.min())
            df_resumen_coef["AOR_Max"] = np.exp(coefs.max())

        return df_resumen_coef

    def pseudo_r2(self):
        """R² de Cox & Snell y de Nagelkerke en cada pliegue.

        Solo aplica si el último paso del Pipeline es una regresión
        logística. Se calculan fuera de muestra: cada modelo de pliegue
        sobre los datos que dejó afuera, con el modelo nulo igual a la tasa
        del objetivo en ese pliegue.

        Cautela: con class_weight="balanced" las probabilidades se corren
        hacia 0.5, así que la verosimilitud sobre la mezcla real de clases
        baja y estos R² salen bajos (incluso negativos). Eso no significa
        que el modelo ordene mal los hogares; eso lo mide el AP.
        """
        self._requiere_cv()
        final = (
            self.pipeline[-1]
            if hasattr(self.pipeline, "named_steps")
            else self.pipeline
        )
        if not isinstance(final, LogisticRegression):
            raise ValueError(
                "Los pseudo-R² solo aplican cuando el modelo es una "
                "regresión logística."
            )

        filas = []
        for estimador, idx_te in zip(
            self.scores["estimator"], self.scores["indices"]["test"]
        ):
            x_te = self.X_train.iloc[idx_te]
            y_te = self.y_train.iloc[idx_te]
            filas.append(
                {
                    "Cox_Snell": _cox_snell_r2(estimador, x_te, y_te),
                    "Nagelkerke": _nagelkerke_r2(estimador, x_te, y_te),
                }
            )

        df_r2 = pd.DataFrame(filas)
        df_r2.index.name = "fold"
        # Mismo formato que resumen_metricas (np.std, ddof=0)
        for col, val in df_r2.items():
            print(f"{col}: {np.mean(val):.3f} ± {np.std(val):.3f}")
        return df_r2

    def bootstrap_ic(
        self, umbral=None, n_boot=1000, alpha=0.05, random_state=None
    ):
        """Intervalo de confianza empírico por bootstrap de conglomerados.

        Junta las predicciones fuera de muestra de todos los pliegues y
        remuestrea grupos completos (por ejemplo, SEGMENTO) con reemplazo:
        los hogares de un mismo segmento están correlacionados y el
        segmento es la unidad de muestreo y de los pliegues. En cada
        réplica se recalculan accuracy, precision, recall y kappa (con el
        umbral) y average precision; el IC son los percentiles alpha/2 y
        1 - alpha/2 de esa distribución empírica.

        Parámetros
        ----------
        umbral       : umbral de decisión. Por defecto, el de la clase
        n_boot       : número de réplicas bootstrap (default 1000)
        alpha        : 1 - nivel de confianza (default 0.05, IC del 95 %)
        random_state : semilla. Por defecto, la de la clase. Con la misma
                       semilla, grupos y pliegues, todos los modelos
                       reciben las mismas réplicas (comparación pareada)

        Devuelve un DataFrame por métrica con la estimación sobre todas
        las predicciones juntas (puede diferir un poco de la media entre
        pliegues) y los límites del IC. La distribución queda en
        self.bootstrap. Solo se remuestrean las predicciones, no se
        reentrena: el IC refleja la incertidumbre de la muestra de
        evaluación, no la del entrenamiento.

        Es decir crea un vector de predicciones global y hace el bootstrap
        sobre esos vectores, luego en cada iteración saca valores de ese
        vector tomando en cuenta los grupos y en base a los vectores
        extraídos se calculan métricas, de ese proceso nacen las
        distribuciones empíricas que dan pie a los IC.

        Este enfoque se toma en lugar de reentrenar una y otra vez pues con
        las Pipelines desarrolladas esto sería muy lento y se conservan las
        ventajas de aumentar la confianza en las estimaciones al eliminar
        ruido de tomar solo un corte.
        """
        self._requiere_cv()
        umbral = self.umbral if umbral is None else umbral
        if random_state is None:
            random_state = self.random_state

        # Predicciones fuera de muestra en el orden de los pliegues
        y = np.concatenate([np.asarray(y_te) for y_te, _ in self.probas_cv])
        y = y.astype(int)
        proba = np.concatenate([p for _, p in self.probas_cv])
        pred = (proba >= umbral).astype(int)
        posiciones = np.concatenate(self.scores["indices"]["test"])
        codigos, grupos_unicos = pd.factorize(
            np.asarray(self.grupos)[posiciones]
        )
        n_grupos = len(grupos_unicos)
        # Celda de la matriz de confusión de cada fila: 0=TN, 1=FP, 2=FN,
        # 3=TP
        celda = 2 * y + pred

        def metricas(pesos):
            tn, fp, fn, tp = np.bincount(celda, weights=pesos, minlength=4)
            n = tn + fp + fn + tp
            p_o = (tp + tn) / n
            # Acuerdo esperado por azar, a partir de las marginales
            p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
            usar = pesos > 0
            return {
                "accuracy": p_o,
                "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
                "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
                "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0,
                "average_precision": average_precision_score(
                    y[usar], proba[usar], sample_weight=pesos[usar]
                ),
            }

        # Cada réplica como pesos: cuántas veces salió el grupo de cada
        # fila (equivale a repetir sus filas, sin copiarlas)
        rng = np.random.default_rng(random_state)
        replicas = []
        for _ in range(n_boot):
            conteo = np.bincount(
                rng.integers(0, n_grupos, n_grupos), minlength=n_grupos
            )
            replicas.append(metricas(conteo[codigos].astype(float)))
        self.bootstrap = pd.DataFrame(replicas)

        estimacion = pd.Series(metricas(np.ones(len(y))))
        return pd.DataFrame(
            {
                "estimacion": estimacion,
                "ic_inf": self.bootstrap.quantile(alpha / 2),
                "ic_sup": self.bootstrap.quantile(1 - alpha / 2),
            }
        ).rename_axis("metrica")

    def curvas_pr(self, thresholds_to_show=(0.2, 0.4, 0.6, 0.8), ax=None):
        """Curva precisión-recall de cada pliegue con los umbrales k
        marcados."""
        self._requiere_cv()
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        # Probabilidades ya calculadas por pliegue (sin reentrenar)
        for fold, (y_te, y_proba) in enumerate(self.probas_cv):
            # 1. Calcular precisión, recall y umbrales (thresholds)
            precision, recall, thresholds = precision_recall_curve(
                y_te, y_proba
            )
            ap = average_precision_score(y_te, y_proba)

            # 2. Graficar la curva PR del fold
            (line,) = ax.plot(
                recall,
                precision,
                alpha=0.5,
                label=f"Fold {fold + 1} (AP = {ap:.3f})",
            )
            fold_color = line.get_color()

            # 3. Marcar los umbrales k en la curva
            for k in thresholds_to_show:
                # Encontrar el índice del umbral más cercano a k
                idx = np.argmin(np.abs(thresholds - k))
                r_k, p_k = recall[idx], precision[idx]

                # Dibujar punto sobre la curva
                ax.scatter(r_k, p_k, color=fold_color, s=40, zorder=5)

                # Etiquetar el valor de k (solo en el Fold 1, para no
                # saturar)
                if fold == 0:
                    ax.annotate(
                        f"k={k}",
                        (r_k, p_k),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        alpha=0.8,
                    )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            "Curvas Precision-Recall por Fold con Puntos de Umbral (k)"
        )
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)
        return ax

    def curva_umbral(self, k_thresholds=None, ax=None):
        """Precisión y recall promedio entre pliegues según el umbral k.

        Guarda en self.k_intersect el umbral donde se cruzan.
        """
        self._requiere_cv()
        # 1. Definir una malla de umbrales k desde 0 hasta 1
        if k_thresholds is None:
            k_thresholds = np.linspace(0.01, 0.99, 100)

        # Listas para almacenar las métricas por fold
        precisions_per_k = []
        recalls_per_k = []

        # 2. Iterar sobre las probabilidades de cada fold (sin reentrenar)
        for y_te, y_proba in self.probas_cv:
            # Evaluar Precision y Recall para cada k en la malla
            fold_precisions = []
            fold_recalls = []

            for k in k_thresholds:
                y_pred_k = (y_proba >= k).astype(int)

                # Calcular métricas para el umbral k
                tp = np.sum((y_te == 1) & (y_pred_k == 1))
                fp = np.sum((y_te == 0) & (y_pred_k == 1))
                fn = np.sum((y_te == 1) & (y_pred_k == 0))

                prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                fold_precisions.append(prec)
                fold_recalls.append(rec)

            precisions_per_k.append(fold_precisions)
            recalls_per_k.append(fold_recalls)

        # 3. Promediar el comportamiento a través de los folds
        mean_precision = np.mean(precisions_per_k, axis=0)
        std_precision = np.std(precisions_per_k, axis=0)

        mean_recall = np.mean(recalls_per_k, axis=0)
        std_recall = np.std(recalls_per_k, axis=0)

        # 4. Graficar Precision y Recall vs Umbral k
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        # Curva de Precision
        ax.plot(
            k_thresholds,
            mean_precision,
            label="Precision Promedio",
            color="#1f77b4",
            lw=2,
        )
        ax.fill_between(
            k_thresholds,
            np.clip(mean_precision - std_precision, 0, 1),
            np.clip(mean_precision + std_precision, 0, 1),
            color="#1f77b4",
            alpha=0.15,
        )

        # Curva de Recall
        ax.plot(
            k_thresholds,
            mean_recall,
            label="Recall Promedio",
            color="#ff7f0e",
            lw=2,
        )
        ax.fill_between(
            k_thresholds,
            np.clip(mean_recall - std_recall, 0, 1),
            np.clip(mean_recall + std_recall, 0, 1),
            color="#ff7f0e",
            alpha=0.15,
        )

        # Marcar el punto donde se intersectan (Trade-off equilibrado)
        idx_intersection = np.argmin(np.abs(mean_precision - mean_recall))
        k_intersect = k_thresholds[idx_intersection]
        val_intersect = mean_precision[idx_intersection]
        self.k_intersect = k_intersect

        ax.axvline(x=k_intersect, color="grey", linestyle="--", alpha=0.7)
        ax.scatter([k_intersect], [val_intersect], color="red", zorder=5)
        ax.annotate(
            f"Cruce (k ≈ {k_intersect:.2f})",
            (k_intersect, val_intersect),
            xytext=(15, -15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

        ax.set_xlabel("Umbral de Decisión (k)")
        ax.set_ylabel("Puntaje")
        ax.set_title(
            "Compromiso (Trade-off) Precision vs Recall según el Umbral k"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.figure.tight_layout()
        return ax

    def comparar_train_test(self, ax=None):
        """Media ± desviación estándar de cada métrica en train y en test
        entre pliegues, para ver el sobreajuste.

        Las métricas de train salen de los modelos que ya ajustó
        evaluar_cv(), evaluados sobre las filas con las que se entrenó
        cada pliegue, con los mismos scorers y el mismo umbral (sin
        reentrenar). Los muestreadores (SMOTE, ADASYN, submuestreo) solo
        actúan en fit, así que train se mide sobre las filas originales.

        Guarda la tabla en self.train_test: medias, desviaciones y la
        brecha (train - test) por métrica.
        """
        self._requiere_cv()
        scoring = _crear_scoring(self.umbral)
        metricas = list(scoring)

        # 1. Métricas de train por pliegue con los modelos ya ajustados
        filas_train = []
        for estimador, idx_tr in zip(
            self.scores["estimator"], self.scores["indices"]["train"]
        ):
            x_tr = self.X_train.iloc[idx_tr]
            y_tr = self.y_train.iloc[idx_tr]
            filas_train.append(
                {
                    m: get_scorer(s)(estimador, x_tr, y_tr)
                    for m, s in scoring.items()
                }
            )
        df_train = pd.DataFrame(filas_train)

        # 2. Resumen entre pliegues (np.std, ddof=0, como en
        # resumen_metricas)
        test = {m: np.asarray(self.scores[f"test_{m}"]) for m in metricas}
        tabla = pd.DataFrame(
            {
                "train_media": [df_train[m].mean() for m in metricas],
                "train_std": [np.std(df_train[m]) for m in metricas],
                "test_media": [test[m].mean() for m in metricas],
                "test_std": [np.std(test[m]) for m in metricas],
            },
            index=pd.Index(metricas, name="metrica"),
        )
        tabla["brecha"] = tabla["train_media"] - tabla["test_media"]
        self.train_test = tabla

        # 3. Barras agrupadas: train y test por métrica
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        x = np.arange(len(metricas))
        ancho = 0.38
        for desplazamiento, conjunto, color in (
            (-ancho / 2, "train", "#1f77b4"),
            (ancho / 2, "test", "#ff7f0e"),
        ):
            barras = ax.bar(
                x + desplazamiento,
                tabla[f"{conjunto}_media"],
                ancho,
                yerr=tabla[f"{conjunto}_std"],
                capsize=4,
                color=color,
                alpha=0.85,
                label=f"{conjunto.capitalize()} (media ± std)",
            )
            ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)

        # El kappa puede ser negativo: el eje baja si hace falta
        minimo = (
            tabla[["train_media", "test_media"]].to_numpy()
            - tabla[["train_std", "test_std"]].to_numpy()
        ).min()
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, metricas)
        ax.set_xlabel("Métrica")
        ax.set_ylabel("Puntaje")
        ax.set_title("Métricas promedio entre pliegues: train vs test")
        ax.set_ylim(min(0, minimo - 0.05), 1.05)
        ax.legend(loc="best")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.figure.tight_layout()
        return ax

    def matriz_confusion(self, ax=None):
        """Matriz de confusión de las predicciones de validación cruzada
        con el umbral de la clase (sin reentrenar)."""
        self._requiere_cv()
        y_true = np.concatenate([y_te for y_te, _ in self.probas_cv])
        y_pred = np.concatenate(
            [y_proba >= self.umbral for _, y_proba in self.probas_cv]
        ).astype(int)

        # 2. Generar la matriz en texto/array
        cm = confusion_matrix(y_true, y_pred)
        print("Matriz de confusión:\n", cm)

        # 3. Graficar la matriz de forma visual
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", ax=ax)
        return disp.ax_

    def ajustar_final(self):
        """Ajusta una copia del Pipeline sobre todo train y la devuelve.

        Con grid devuelve el GridSearchCV ajustado: busca los
        hiperparámetros en todo train y el Pipeline ganador queda en
        best_estimator_.
        """
        modelo = clone(self._estimador())
        with sklearn.config_context(enable_metadata_routing=True):
            try:
                modelo.fit(self.X_train, self.y_train, groups=self.grupos)
            except TypeError as error:
                # Ningún paso usa los grupos (por ejemplo, un dummy)
                if "not routed to any object" not in str(error):
                    raise
                modelo.fit(self.X_train, self.y_train)
        return modelo
