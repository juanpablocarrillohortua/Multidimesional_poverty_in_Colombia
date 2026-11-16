"""Streamlit app that deploys the poverty model."""

from pathlib import Path

import joblib
import sklearn
import streamlit as st
from app_config import settings
from preprocessing import HouseHold
from pydantic import ValidationError

LABELS = settings.FIELD_LABELS
# ECV codes the model uses for yes/no questions (1 = yes)
YES_NO = {1: "Sí", 2: "No"}
STYLE = Path(__file__).with_name("style.css")


@st.cache_resource
def load_artifact():
    """Load the production model once per process."""
    return joblib.load(settings.MODEL_PATH)


@st.cache_data
def load_css():
    """Stylesheet of the app (style.css next to this file)."""
    return STYLE.read_text(encoding="utf-8")


def probability_meter(proba, threshold, is_poor):
    """Bar with the probability and a tick at the decision threshold."""
    state = "poor" if is_poor else "not-poor"
    return f"""
<div class="meter">
  <div class="meter-label">
    Probabilidad de pobreza multidimensional: {proba:.1%}
  </div>
  <div class="meter-track">
    <div class="meter-fill {state}" style="width: {proba:.2%}"></div>
    <div class="meter-tick" style="left: {threshold:.2%}"></div>
  </div>
  <div class="meter-tick-label">
    <span style="left: {threshold:.2%}">umbral {threshold:.0%}</span>
  </div>
</div>
"""


def count_input(field, min_value=0, value=0, **kwargs):
    """Whole-number input for a HouseHold count."""
    return st.number_input(
        LABELS[field],
        min_value=min_value,
        value=value,
        step=1,
        key=field,
        **kwargs,
    )


def coded_select(field, options):
    """Drop-down that shows each option as 'code — label'."""
    return st.selectbox(
        LABELS[field],
        options=list(options),
        format_func=lambda code: f"{code} — {options[code]}",
        key=field,
    )


st.set_page_config(
    page_title="Pobreza multidimensional",
    page_icon="🏠",
    layout="centered",
)
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
st.title("Pobreza multidimensional en Colombia")
st.caption(
    "Estima la probabilidad de que un hogar urbano sea pobre "
    "multidimensional (IPM) a partir de las condiciones del hogar y de la "
    "vivienda."
)

artifact = load_artifact()
if artifact["sklearn_version"] != sklearn.__version__:
    st.warning(
        f"El modelo se guardó con scikit-learn "
        f"{artifact['sklearn_version']} y está instalada la versión "
        f"{sklearn.__version__}: las predicciones pueden variar."
    )

answers = {}

# One card per section: related questions stay together
with st.container(border=True, key="card-household"):
    st.subheader("Composición del hogar")
    col_1, col_2 = st.columns(2)
    with col_1:
        answers["members"] = count_input("members", min_value=1, value=1)
        answers["members_6_plus"] = count_input("members_6_plus", value=1)
    with col_2:
        answers["members_15_plus"] = count_input("members_15_plus", value=1)
        answers["members_12_17"] = count_input("members_12_17")

with st.container(border=True, key="card-people"):
    st.subheader("Educación, salud y trabajo infantil")
    col_1, col_2 = st.columns(2)
    with col_1:
        answers["illiterate"] = count_input("illiterate")
        answers["uninsured"] = count_input("uninsured")
    with col_2:
        # Without members aged 12 to 17 the question doesn't apply: it is
        # blocked and 0 is sent
        no_teens = answers["members_12_17"] == 0
        working = count_input(
            "working_12_17",
            disabled=no_teens,
            help="No aplica: no hay personas de 12 a 17 años."
            if no_teens
            else None,
        )
        answers["working_12_17"] = 0 if no_teens else working

with st.container(border=True, key="card-housing"):
    st.subheader("Vivienda y servicios")
    col_1, col_2 = st.columns(2)
    with col_1:
        answers["sanitary_service"] = coded_select(
            "sanitary_service", settings.SANITARY_SERVICES
        )
        answers["has_sewerage"] = coded_select("has_sewerage", YES_NO) == 1
    with col_2:
        answers["water_source"] = coded_select(
            "water_source", settings.WATER_SOURCES
        )
        answers["has_aqueduct"] = coded_select("has_aqueduct", YES_NO) == 1

# Validate on every rerun: any error blocks the button
try:
    household = HouseHold(**answers)
    errors = []
except ValidationError as exc:
    household = None
    errors = [
        error["msg"].removeprefix("Value error, ") for error in exc.errors()
    ]

for message in errors:
    st.error(message)

predict = st.button(
    "Predecir", type="primary", disabled=bool(errors), width="stretch"
)

if predict and household is not None:
    row = household.create_input_row()
    proba = float(artifact["pipeline"].predict_proba(row)[0, 1])
    is_poor = proba >= settings.UMBRAL

    st.subheader("Resultado")
    st.markdown(
        probability_meter(proba, settings.UMBRAL, is_poor),
        unsafe_allow_html=True,
    )
    if is_poor:
        st.error(
            f"Hogar clasificado como **pobre multidimensional** "
            f"(probabilidad {proba:.1%} ≥ umbral {settings.UMBRAL:.0%})."
        )
    else:
        st.success(
            f"Hogar clasificado como **no pobre multidimensional** "
            f"(probabilidad {proba:.1%} < umbral {settings.UMBRAL:.0%})."
        )
    st.caption(
        "El umbral prioriza detectar hogares pobres (recall) a costa de "
        "algunas falsas alarmas."
    )

    st.subheader("Fila de entrada al modelo")
    st.dataframe(row, hide_index=True)
    st.caption(
        "Los valores vacíos (NaN) los imputa el pipeline: sin personas de "
        "12 a 17 años se activa el indicador de nulidad."
    )
