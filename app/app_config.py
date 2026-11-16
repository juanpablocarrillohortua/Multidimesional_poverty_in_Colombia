"""Configuration for the app."""

from pathlib import Path

from pydantic import PostgresDsn, computed_field
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # pylint: disable=too-few-public-methods
    """App configuration."""

    # Same order as the pipeline's feature_names_in_ (sklearn requires it)
    MODEL_VARS: list = [
        "ORG_analfabetismo",
        "ORG_trabajo_infantil",
        "ORG_sin_aseguramiento",
        "ORG_acueducto",
        "ORG_fuente_agua_cocinar",
        "ORG_alcantarillado",
        "ORG_servicio_sanitario",
    ]

    # Codes the model was trained with (labels from the DANE questionnaire)
    SANITARY_SERVICES: dict[int, str] = {
        1: "Inodoro conectado a alcantarillado",
        2: "Inodoro conectado a pozo séptico",
        3: "Inodoro sin conexión",
        4: "Letrina",
        5: "Inodoro con descarga directa a fuentes de agua (bajamar)",
        6: "No tiene servicio sanitario",
    }

    # Ordinal code from best to worst (FUENTE_AGUA_ORDINAL in
    # eda_consolidate.ipynb), not the DANE code of P8530
    WATER_SOURCES: dict[int, str] = {
        1: "Acueducto público",
        2: "Acueducto comunal o veredal",
        3: "Agua embotellada o en bolsa",
        4: "Pozo con bomba",
        5: "Pila pública",
        6: "Carrotanque",
        7: "Aguatero",
        8: "Pozo sin bomba, aljibe, jagüey o barreno",
        9: "Agua lluvia",
        10: "Río, quebrada, manantial o nacimiento",
    }

    UMBRAL: float = 0.45

    # Saved production model, relative to this file so it doesn't depend
    # on the folder `streamlit run` is launched from
    MODEL_PATH: Path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "multidimensional_poverty_model_v1.joblib"
    )

    # Question shown for each HouseHold field (form labels and errors)
    FIELD_LABELS: dict[str, str] = {
        "members": "Personas en el hogar",
        "members_15_plus": "Personas de 15 años o más",
        "illiterate": (
            "Personas de 15 años o más que no saben leer ni escribir"
        ),
        "members_6_plus": "Personas de 6 años o más",
        "uninsured": "Personas de 6 años o más sin afiliación a salud",
        "members_12_17": "Personas de 12 a 17 años",
        "working_12_17": "Personas de 12 a 17 años que trabajan",
        "sanitary_service": "Tipo de servicio sanitario",
        "has_sewerage": "¿La vivienda tiene alcantarillado?",
        "has_aqueduct": "¿La vivienda tiene acueducto?",
        "water_source": "Fuente del agua para cocinar",
    }


settings = Settings()
