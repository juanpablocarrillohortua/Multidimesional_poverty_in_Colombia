# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine Learning course project: predict household multidimensional poverty
in Colombia from DANE's *Encuesta Nacional de Calidad de Vida* (ECV) 2025
microdata. The target is built with the Alkire-Foster double cutoff, replicating
DANE's official IPM and the robustness analyses of Zhao, Cheah & Moses (2024),
PLOS ONE 19(5): e0300263, whose PDF is in `docs/`.

Pipeline, in order:

1. `domain_ultis/unzip_year.py` extracts the raw zips (see ECV data below).
2. `notebooks/eda_consolidate.ipynb` is the main notebook. It merges every ECV
   table into one person-level frame (235,350 × 902), filters to cities
   (`CLASE == 1`), runs `domain.ipm.IPM` to get one row per household with the
   `POBRE` target, then starts feature EDA and imputation.
3. `notebooks/consolidate_dfs.ipynb` and `full_analisis.ipynb` are earlier
   explorations that worked out the join keys. They are reference material,
   not part of the pipeline.

`config.py` defines `config.WORKDIR`, `config.DATADIR` (`data/`) and
`config.EXTRACTED` (`data/extracted/`). `Dockerfile`, `.dockerignore` and
`.env` are empty placeholders. Generated figures and tables go in
`outputs/plots/` and `outputs/tables/`.

## Environment and commands

Windows, Python 3.13 venv at `.venv/` (`.venv/Scripts/python.exe`). `make` is
installed, and the Makefile uses the venv interpreter automatically.

```bash
make install        # runtime deps from requirements.txt
make install-dev    # ruff==0.16.7 from requirements/dev.txt
make lint           # ruff check utils/ and notebooks/ (lint-py / lint-nb for one side)
make format         # ruff --fix (never deletes code) + ruff format
make quality        # clean + lint + format-check: read-only gate, same as CI
make clean          # tools/clean.py: __pycache__, .ipynb_checkpoints, tool caches
.venv/Scripts/python.exe -m ruff check utils/fancy_hist.py   # a single file
python domain_ultis/unzip_year.py data/ECV_2025              # extract a survey year
```

`requirements.txt` pins pandas 3.0, numpy, scikit-learn, scipy, statsmodels,
matplotlib, seaborn and Jupyter. It was frozen on another machine, so the local
venv can fall behind it: if `domain.ipm` or a `utils/` module raises
`ModuleNotFoundError`, run `make install`. `plotly`, which `utils/fast_OLS.py`
needs, is not in it. pandas 3.0 has Copy-on-Write always on (chained assignment
does not modify the frame) and a default `str` dtype for text columns.

`make quality` runs in CI on every push and pull request to `main`
(`.github/workflows/quality.yml`): `ubuntu-latest`, Python 3.13, ruff only.
It gates **only `utils/` and `notebooks/`**. `domain/`, `domain_ultis/` and
`config.py` are not linted, and `domain/` does not pass the 79-column rule
(`ipm.py` is formatted at 88).

**Scaffolding copied from other projects. Do not rely on it:**
- `make test` / `make validate` fail: there is no `tests/` directory and pytest is not installed.
- `make scrape` / `scrape-venta` (FincaRaiz scraper), the `data_loader/` exclusion in `ruff.toml` (LUNA25) and the LUNA25 column names in `utils/README.md` examples all come from earlier projects.

## Lint rules (ruff.toml)

79-column lines, `py313`, rule sets E/W/F/I/N. Notebooks are linted cell by cell
with E402 and F401 ignored. F401/F811/F841 are marked unfixable, so `make format`
reports unused imports and variables but never removes them. Uppercase math
names (`X`, `W`, `M`, `R`, `B`, `X_with_const`, `invW_T`) are exempt from
pep8-naming.

## ECV data (`data/`, gitignored)

`data/ECV_2025/` holds one zip per survey module, each with the same table as
`.csv`, `.DTA` and `.sav`. `data/ECV_2023/` is empty. Notebooks don't read the
zips: `unzip_year.py` flattens them into
`data/extracted/ECV_2025_extracted/<ext>/` (fixing cp437-mangled accented
names), and notebooks read `config.EXTRACTED / "ECV_2025_extracted" / "csv"`.
The filenames there are the Spanish table titles with accents
(`"Educación.csv"`, `"Adquisición de bienes y servicios por ítem.CSV"`). The
`FILES` dict in `eda_consolidate.ipynb` maps short aliases to them.

CSV format: UTF-8, `sep=";"`, empty fields are skipped questions. Decimals use
`,` in every file **except** `Adquisicion_bienes_servicios_item`, which uses
`.`. Read `P1_DEPARTAMENTO` and `P1_MUNICIPIO` as `str` (zero-padded codes;
`DIVIPOLA` = 2 + 3 digits). The Educación table has some lowercase column names
(`Directorio`, `P8587s1`), so upper-case all columns before merging.
Variables are DANE questionnaire codes (`P6040`, `P5000`, …). Their questions
are in `domain/variables_naming.py`.

### Join keys: the main trap

Every table has the same key columns: `DIRECTORIO`, `SECUENCIA_ENCUESTA`,
`SECUENCIA_P`, `ORDEN`, `FEX_C`. **Their meaning changes with the level of the
table.** `SECUENCIA_P` points to the parent record's `SECUENCIA_ENCUESTA`. A
merge on the shared column names, or `pd.merge` with no `on=`, joins the wrong
rows without raising an error.

| Level | Unique key | Tables (2025 row counts) |
| --- | --- | --- |
| vivienda | `DIRECTORIO` | Datos_vivienda (86,848) |
| hogar | `DIRECTORIO` + `SECUENCIA_ENCUESTA` (`SECUENCIA_P` is always 1) | Servicios_hogar, Condiciones_vida_hogar_tenencia_bienes, Tenencia_financiacion_vivienda_ocupa_hogar, Adquisicion_bienes_servicios (87,060 each) |
| persona | `DIRECTORIO` + `SECUENCIA_P` + `ORDEN` | Caracteristicas_composicion_hogar, Salud, Variables_diseno_muestral (235,350); Educacion, TIC (221,043); Fuerza_trabajo (195,074); Trabajo_infantil (25,969); Atencion_integral <5 años (14,307) |
| item (child of hogar) | `DIRECTORIO` + `SECUENCIA_P` + `P3204` | Adquisicion_bienes_servicios_item (1,473,207) |

The recipe `eda_consolidate.ipynb` uses (follow it rather than re-deriving):
- Base: Caracteristicas. Left-join each person module on
  `(DIRECTORIO, SECUENCIA_P, ORDEN)` with `validate="1:1"`, dropping
  `SECUENCIA_ENCUESTA` (it repeats `ORDEN` at this level).
- Hogar tables: drop `SECUENCIA_P`/`ORDEN`, rename `SECUENCIA_ENCUESTA` →
  `SECUENCIA_P`, join on `(DIRECTORIO, SECUENCIA_P)` with `validate="m:1"`.
- Vivienda: drop the three sequence columns, join on `DIRECTORIO`, `m:1`.
- Always pass `validate=`, so a wrong key raises instead of multiplying rows,
  and check that the result keeps 235,350 rows.
- `FEX_C` (expansion factor) differs in its last decimal between tables for the
  same record. Never join on it. Keep it from Variables_diseno_muestral only.
- The small `(programas)` and `(personas_escritura)` sub-tables use the same
  parent-pointer scheme. Check uniqueness before joining them.

In the merged frame, a `NaN` from a person module means the module didn't apply
to that person (age filter), not a missing answer. `BLOQUE` is the only text
column. It is a survey-design variable (calibration domain), not a predictor.

## `domain/`: target construction

`domain/ipm.py` is the core of the project. `IPM` is a scikit-learn
`BaseEstimator`/`TransformerMixin` that takes **one row per person** with raw
upper-case ECV codes and returns **one row per household**: keys, `PESO` (sum of
`FEX_C`), one 0/1 column per cutoff indicator, `CAR_*` / `ORG_*` features, the
weighted score `C` and the target `POBRE`. Things to know:

- Indicators are data, not code: `DANE_INDICADORES` gives each one a dimension,
  a person filter, a variable, an operator and a threshold `z`, any of which may
  be a `{CLASE: value}` dict. Derived person-level variables (`ANOS_EDUC`,
  `PEA`, `DLD`, `FORMAL`…) are built in `_preparar`, which documents each DANE
  rule. Change thresholds via `DANE_CODIGOS` / the `codigos` param.
- **Modified scheme** used by the notebook: `IPM(caracteristicas=INDICADORES_ELIMINADOS)`
  moves 6 of the 15 indicators out of the cutoff, so they can be features
  (`CAR_*` binary, `ORG_*` original values or eligible-member proportions).
  Weights are recomputed equally per dimension over the remaining 9.
- **Leakage**: the lists `VARIABLES_CORTE` (raw variables that build the
  target, so drop them from X), `VARIABLES_COMPARTIDAS` (feed both a removed
  indicator and the cutoff, so don't use them raw), `VARIABLES_LIBERADAS` (safe
  to use) and `CARACTERISTICAS_EXCLUIDAS` (`CAR_trabajo_informal` is a target
  proxy) define what may enter the model.
- It changes the row unit, so it can't sit inside a `Pipeline` before a model.
  Run it once before splitting (`fit` learns nothing from X), then split by
  household or `SEGMENTO` (e.g. `GroupKFold`).
- Robustness analyses are static methods on the output table: `dominancia_k`
  (paper fig. 2), `esquemas_sin_indicador` / `esquemas_peso_dimension` +
  `kendall_esquemas` (tables 3 and 6), with `grafico_*` plotters. They
  recompute `C`/`POBRE` from `w` and `k` without re-running `transform`.
- `transform` fails if columns are missing, lowercase or carry `_x`/`_y` merge
  suffixes. It needs the raw `P…` codes, not the long names the notebook's
  `build_rename` produces (that goes into a separate `df_renamed`).

`domain/variables_naming.py`: `VARIABLES` maps 654 ECV codes to
`{pregunta, archivo, nivel}` (`archivo` is a DANE file id such as `F24`, see
`ARCHIVOS`), plus `variables_ipm_dane`. Its module docstring is stale: it
describes `FLUJO`, `IndiceAF`, `IPM_DANE`, `particion()` and a JSON export
that don't exist.

`domain/` doesn't follow the `utils/` rules below: identifiers are Spanish,
plotters return a `Figure` rather than accept `ax`, and it imports sklearn and
scipy at module level.

## `utils/` analysis toolkit

`utils/README.md` is the full API reference (parameters, return values, which
statistical test is chosen). Update it when you change a module. Its
"Installed / imports cleanly" section is stale. Architectural rules:

- One self-contained module per analysis idea. Modules **never import each other** and `__init__.py` re-exports nothing. A helper needed by two modules is copied, not shared.
- Notebooks import `utils`, `domain` and `config` by putting the repo root on the path: `sys.path.insert(0, str(Path.cwd().parent))`, then `from utils.fancy_barplot import fancy_bars`.
- Plotting contract (used by `fancy_barplot`, `fancy_hist`, `segmented_bars`; follow it in new modules): accept `ax=None`, return the Axes, don't call `plt.show()`, and `title=None` builds the title from the column name. `bivaraite_boxplot` and `category_vs_numeric_plot` predate this contract: they call `plt.show()` and can't go into subplot grids.
- Identifiers are in English. Figure text and `ValueError` messages are in Spanish.
- `mutual_inf_plot` functions default to `target="is_popular"` (from an earlier project), so always pass `target` explicitly.

## Writing a new `utils/` module

Match the existing house style. `utils/fancy_barplot.py` and `utils/triple_plot.py` are the reference implementations:

- **Layout**: module docstring → style constants (`INK, TXT, GRID`, `PALETTE`, `C_DARK`…) → private `_`-prefixed helpers → public function last.
- **Signature**: `(df, <column names>, ..., title=None, palette=None, ax=None, figsize=(w, h))`, returning `plt.Axes`. Build the figure only when `ax is None`, so callers can compose subplots.
- **`title=None` means auto-generate** from the column name via `_pretty()` (`"foo_bar"` → `"Foo Bar"`).
- **Validate literal-string arguments explicitly** and `raise ValueError` with a Spanish message. See [fancy_barplot.py:107](utils/fancy_barplot.py#L107).
- **Language split** (stated in [triple_plot.py:3-6](utils/triple_plot.py#L3-L6)): text drawn on the figure and user-facing docstrings in Spanish, code identifiers in English.
- **Reuse rather than re-derive**: `_pretty()`, `_colors()`, `_style()`, `_bar_labels()` in [fancy_barplot.py](utils/fancy_barplot.py); `_fd_bins()` (Freedman–Diaconis bin count) and `_hist_density()` in [triple_plot.py:101](utils/triple_plot.py#L101). Since modules don't cross-import, copy the helper if you need it elsewhere.
