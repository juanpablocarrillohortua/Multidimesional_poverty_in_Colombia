"""Preprocess the household inputs for the model.

The user only enters data about the household; this module computes
the proportions and creates the indicators the model needs.

The UI asks for:

- number of members in the household;
- number of members aged 15 or older;
- number of illiterate members (out of those aged 15 or older);
- number of members aged 6 or older;
- number of members without health insurance (out of those aged 6 or
  older);
- number of members aged 12 to 17 (if it is 0, the next question is
  blocked too; otherwise an error is raised);
- number of members aged 12 to 17 who work;
- type of sanitary service;
- whether there is access to sewerage;
- whether there is access to an aqueduct;
- type of water used for cooking.

The module implements a pydantic class that returns a one-row
``pd.DataFrame``, just as the sklearn pipeline expects.

It also checks that no input is larger than the number of members in
the household; if one is, it raises an error and stops the row from
being created.
"""

import numpy as np
import pandas as pd
from app_config import settings
from pydantic import BaseModel, Field, model_validator

# (part, group): the count `part` is a subset of the members in `group`
_SUBSETS = [
    ("members_15_plus", "members"),
    ("members_6_plus", "members"),
    ("members_12_17", "members_6_plus"),
    ("illiterate", "members_15_plus"),
    ("uninsured", "members_6_plus"),
    ("working_12_17", "members_12_17"),
]


def _share(part, group):
    """Proportion of `group` that `part` represents; NaN if `group` is 0."""
    return part / group if group > 0 else np.nan


class HouseHold(BaseModel):
    """Answers of the form for one household."""

    members: int = Field(ge=1, description="Members in the household.")
    members_15_plus: int = Field(ge=0, description="Members aged 15+.")
    illiterate: int = Field(
        ge=0, description="Members aged 15+ who can't read and write."
    )
    members_6_plus: int = Field(ge=0, description="Members aged 6+.")
    uninsured: int = Field(
        ge=0, description="Members aged 6+ without health insurance."
    )
    members_12_17: int = Field(ge=0, description="Members aged 12 to 17.")
    working_12_17: int = Field(
        default=0, ge=0, description="Members aged 12 to 17 who work."
    )
    sanitary_service: int = Field(
        ge=min(settings.SANITARY_SERVICES),
        le=max(settings.SANITARY_SERVICES),
        description="Code of settings.SANITARY_SERVICES.",
    )
    has_sewerage: bool = Field(description="Access to sewerage.")
    has_aqueduct: bool = Field(description="Access to an aqueduct.")
    water_source: int = Field(
        ge=min(settings.WATER_SOURCES),
        le=max(settings.WATER_SOURCES),
        description="Code of settings.WATER_SOURCES.",
    )

    @model_validator(mode="after")
    def _check_counts(self):
        """No count can be larger than the group it belongs to."""
        for part, group in _SUBSETS:
            n_part, n_group = getattr(self, part), getattr(self, group)
            if n_part > n_group:
                labels = settings.FIELD_LABELS
                raise ValueError(
                    f"«{labels[part]}» ({n_part}) no puede ser mayor que "
                    f"«{labels[group]}» ({n_group})"
                )
        return self

    def create_input_row(self) -> pd.DataFrame:
        """One-row DataFrame with the columns the pipeline expects."""
        row = {
            # NaN without members aged 15+, as in training (the pipeline
            # imputes 0)
            "ORG_analfabetismo": _share(self.illiterate, self.members_15_plus),
            # NaN without members aged 12 to 17: the pipeline imputes 0
            # and sets missingindicator_ORG_trabajo_infantil to 1
            "ORG_trabajo_infantil": _share(
                self.working_12_17, self.members_12_17
            ),
            # The pipeline doesn't impute this column (it never had NaN in
            # training), and the IPM counts a household without members
            # aged 6+ as not deprived, so it gets 0
            "ORG_sin_aseguramiento": (
                _share(self.uninsured, self.members_6_plus)
                if self.members_6_plus > 0
                else 0.0
            ),
            # 1 = yes, 2 = no, as in the ECV
            "ORG_acueducto": 1 if self.has_aqueduct else 2,
            "ORG_fuente_agua_cocinar": self.water_source,
            "ORG_alcantarillado": 1 if self.has_sewerage else 2,
            "ORG_servicio_sanitario": self.sanitary_service,
        }
        return pd.DataFrame([row], columns=settings.MODEL_VARS)
