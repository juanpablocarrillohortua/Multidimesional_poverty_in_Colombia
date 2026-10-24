"""
Evaluación de un Pipeline de clasificación binaria con validación cruzada
agrupada y estratificada, con un umbral de decisión explícito.

Reúne en una clase el flujo de `notebooks/modeling.ipynb`: validación
cruzada, resumen de métricas, frecuencia de selección de variables, curvas
precisión-recall, compromiso precisión-recall según el umbral, matriz de
confusión y ajuste final sobre todo train.

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

from collections import Counter
from pathlib import Path

import joblib
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
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

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
    random_state : semilla del separador por defecto (default 42)
    pickle       : si es True, evaluar_cv() guarda sus resultados en un
                   .pkl y, en las siguientes ejecuciones, los carga en vez
                   de reentrenar mientras no cambie nada de lo que los
                   determina (Pipeline, datos, grupos, umbral, pliegues,
                   versión de scikit-learn o el código de este módulo)
    dir_cache    : carpeta de los .pkl. Por defecto data/cache_cv/ en la
                   raíz del repositorio. Cargue solo archivos propios: un
                   pickle puede ejecutar código al abrirse.
    """

    def __init__(
        self,
        pipeline,
        X_train,
        y_train,
        umbral,
        grupos,
        cv=None,
        random_state=42,
        pickle=False,
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
        self.cv = cv
        self.pickle = pickle
        self.dir_cache = Path(dir_cache) if dir_cache else DIR_CACHE
        self.scores = None
        self.probas_cv = None
        self.k_intersect = None

    def _requiere_cv(self):
        if self.scores is None:
            raise ValueError("Primero ejecute evaluar_cv().")

    def _clave_cache(self):
        """Hash de todo lo que determina el resultado de evaluar_cv().

        clone() deja solo los hiperparámetros, así que un Pipeline ya
        ajustado da la misma clave. El código del módulo entra en la clave
        para que cambiar los scorers invalide las cachés anteriores.
        """
        return joblib.hash(
            (
                clone(self.pipeline),
                self.X_train,
                self.y_train,
                self.grupos,
                self.umbral,
                self.cv,
                sklearn.__version__,
                Path(__file__).read_bytes(),
            )
        )

    def evaluar_cv(self):
        """Corre la validación cruzada y guarda las probabilidades por
        pliegue. Devuelve self para poder encadenar llamadas.

        Con pickle=True carga los resultados de la caché si ya existen
        para estos mismos parámetros, sin reentrenar.
        """
        print(f"umbral: {self.umbral}")
        archivo = None
        if self.pickle:
            archivo = self.dir_cache / f"{self._clave_cache()}.pkl"
            if archivo.exists():
                cache = joblib.load(archivo)
                self.scores = cache["scores"]
                self.probas_cv = cache["probas_cv"]
                print(f"Resultados cargados de caché: {archivo}")
                return self

        with sklearn.config_context(enable_metadata_routing=True):
            self.scores = cross_validate(
                estimator=self.pipeline,
                X=self.X_train,
                y=self.y_train,
                params={"groups": self.grupos},
                cv=self.cv,
                scoring=_crear_scoring(self.umbral),
                return_estimator=True,
                return_indices=True,
            )

        # Probabilidades de cada pliegue con los modelos que ya ajustó
        # cross_validate: no se reentrena nada
        self.probas_cv = [
            (
                self.y_train.iloc[idx_te],
                estimador.predict_proba(self.X_train.iloc[idx_te])[:, 1],
            )
            for estimador, idx_te in zip(
                self.scores["estimator"], self.scores["indices"]["test"]
            )
        ]

        if archivo is not None:
            archivo.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"scores": self.scores, "probas_cv": self.probas_cv}, archivo
            )
            print(f"Resultados guardados en caché: {archivo}")
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

        for estimator in self.scores["estimator"]:
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
        if not hasattr(self.scores["estimator"][0]["model"], "coef_"):
            raise ValueError(
                "El modelo no tiene coeficientes (coef_): solo aplica a "
                "modelos lineales."
            )

        registros = []
        n_folds = len(self.scores["estimator"])

        for idx, pipeline in enumerate(self.scores["estimator"]):
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
        """Ajusta una copia del Pipeline sobre todo train y la devuelve."""
        modelo = clone(self.pipeline)
        with sklearn.config_context(enable_metadata_routing=True):
            try:
                modelo.fit(self.X_train, self.y_train, groups=self.grupos)
            except TypeError as error:
                # Ningún paso usa los grupos (por ejemplo, un dummy)
                if "not routed to any object" not in str(error):
                    raise
                modelo.fit(self.X_train, self.y_train)
        return modelo
