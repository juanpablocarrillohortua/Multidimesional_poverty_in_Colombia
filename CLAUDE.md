# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine Learning course project on multidimensional poverty in Colombia, built on
DANE's *Encuesta Nacional de Calidad de Vida* (ECV) microdata. It is at an early
stage: `notebooks/consolidate_dfs.ipynb` (merging the ECV tables) is only started,
and `config.py`, `Dockerfile`, `.dockerignore` and `.env` are empty placeholders.
Generated figures and tables go in `outputs/plots/` and `outputs/tables/`.

## Environment and commands

Windows, Python 3.13 venv at `.venv/` (`.venv/Scripts/python.exe`). `make` is
installed, and the Makefile uses the venv interpreter automatically.

```bash
make lint           # ruff check utils/ and notebooks/ (lint-py / lint-nb for one side)
make format         # ruff --fix (never deletes code) + ruff format
make quality        # clean + lint + format-check: read-only gate
make clean          # tools/clean.py: __pycache__, .ipynb_checkpoints, tool caches
.venv/Scripts/python.exe -m ruff check utils/fancy_hist.py   # a single file
```

`make quality` also runs in CI on every push and pull request to `main`
(`.github/workflows/quality.yml`): `ubuntu-latest`, Python 3.13, ruff from
`requirements/dev.txt`. It gates the same `utils/` and `notebooks/` as locally,
not the root-level modules.

**Scaffolding copied from other projects. Do not rely on it:**
- `make test` / `make validate` fail: there is no `tests/` directory and pytest is not installed.
- `make install` points to `requirements.txt`, which doesn't exist: there is no runtime dependency manifest. `make install-dev` does work, `requirements/dev.txt` pins `ruff==0.16.7` and nothing else.
- `make scrape` / `scrape-venta` (FincaRaiz scraper), the `data_loader/` exclusion in `ruff.toml` (LUNA25) and the LUNA25 column names in `utils/README.md` examples all come from earlier projects.

**The venv only has pandas 3.0, numpy, ipykernel and ruff.** Every `utils/`
module currently fails to import. They need `matplotlib seaborn scipy statsmodels`,
plus `plotly` for `fast_OLS` and `scikit-learn` for `mutual_inf_plot`. The
"Installed / imports cleanly" section of `utils/README.md` is stale. pandas 3.0
has Copy-on-Write always on (chained assignment does not modify the frame) and a
default `str` dtype for text columns.

## Lint rules (ruff.toml)

79-column lines, `py313`, rule sets E/W/F/I/N. Notebooks are linted cell by cell
with E402 and F401 ignored. F401/F811/F841 are marked unfixable, so `make format`
reports unused imports and variables but never removes them. Uppercase math
names (`X`, `W`, `M`, `R`, `B`, `X_with_const`, `invW_T`) are exempt from
pep8-naming. The codebase currently passes `make quality`.

## ECV data (`data/`, gitignored)

`data/ECV_2025/` holds one zip per survey module. Each zip contains the same
table as `.csv`, `.DTA` and `.sav`. `data/ECV_2023/` is empty. Read the CSV
straight from the zip with `zipfile`. The path inside the zip varies (with or
without a folder prefix, `.CSV` in uppercase in the `_item` zip, a `CSV/`
subfolder in the PDET zip, accented filenames), so find it with
`name.lower().endswith(".csv")`, not by exact name.

CSV format: UTF-8, `sep=";"`, CRLF, empty fields are skipped questions.
Decimals use `,` in every file **except**
`Adquisicion_bienes_servicios_item`, which uses `.`. Variables are DANE
questionnaire codes (`P6040`, `P5000`, …). The data dictionary is not in the
repo.

### Join keys: the main trap

Every table has the same key columns: `DIRECTORIO`, `SECUENCIA_ENCUESTA`,
`SECUENCIA_P`, `ORDEN`, `FEX_C`. **Their meaning changes with the level of the
table.** `SECUENCIA_P` points to the parent record's `SECUENCIA_ENCUESTA`. A
merge on the shared column names, or `pd.merge` with no `on=`, joins the wrong
rows without raising an error. Always give explicit `left_on`/`right_on`.

| Level | Unique key | Tables (2025 row counts) |
| --- | --- | --- |
| vivienda | `DIRECTORIO` | Datos_vivienda (86,848) |
| hogar | `DIRECTORIO` + `SECUENCIA_ENCUESTA` (`SECUENCIA_P` is always 1) | Servicios_hogar, Condiciones_vida_hogar_tenencia_bienes, Tenencia_financiacion_vivienda_ocupa_hogar, Adquisicion_bienes_servicios (87,060 each) |
| persona | `DIRECTORIO` + `SECUENCIA_P` + `ORDEN` | Caracteristicas_composicion_hogar, Salud, Variables_diseno_muestral (235,350); Educacion, TIC (221,043); Fuerza_trabajo (195,074); Trabajo_infantil (25,969); Atencion_integral <5 años (14,307) |
| item (child of hogar) | `DIRECTORIO` + `SECUENCIA_P` + `P3204` | Adquisicion_bienes_servicios_item (1,473,207) |

- persona → hogar: `(DIRECTORIO, SECUENCIA_P)` = `(DIRECTORIO, SECUENCIA_ENCUESTA)`.
- hogar → vivienda: `DIRECTORIO`.
- The small `(programas)` and `(personas_escritura)` sub-tables use the same parent-pointer scheme. Check uniqueness before joining them.

The row counts above make good sanity checks after a merge. Person modules
cover only an age-filtered subset, so a left join from Caracteristicas has to
keep 235,350 rows.

`FEX_C` (expansion factor) differs in its last decimal between tables for the
same record. Never join on it. Take it from one table and drop it from the
others. `Factores_expansion_municipios_PDET_2025` is a separate
vivienda-level `FEX_C` for PDET municipalities only (14,900 rows).

## `utils/` analysis toolkit

`utils/README.md` is the full API reference (parameters, return values, which
statistical test is chosen). Update it when you change a module. Architectural rules:

- One self-contained module per analysis idea. Modules **never import each other** and `__init__.py` re-exports nothing. A helper needed by two modules is copied, not shared.
- Notebooks import it by putting the repo root on the path: `sys.path.insert(0, str(Path.cwd().parent))`, then `from utils.fancy_barplot import fancy_bars`.
- Plotting contract (used by `fancy_barplot`, `fancy_hist`, `segmented_bars`; follow it in new modules): accept `ax=None`, return the Axes, don't call `plt.show()`, and `title=None` builds the title from the column name. `bivaraite_boxplot` and `category_vs_numeric_plot` predate this contract: they call `plt.show()` and can't go into subplot grids.
- Identifiers are in English. Figure text and `ValueError` messages are in Spanish.
- `mutual_inf_plot` functions default to `target="is_popular"` (from an earlier project), so always pass `target` explicitly.

## Writing a new `utils/` module

Match the existing house style — `utils/fancy_barplot.py` and `utils/triple_plot.py` are the reference implementations:

- **Layout**: module docstring → style constants (`INK, TXT, GRID`, `PALETTE`, `C_DARK`…) → private `_`-prefixed helpers → public function last.
- **Signature**: `(df, <column names>, ..., title=None, palette=None, ax=None, figsize=(w, h))`, returning `plt.Axes`. Build the figure only when `ax is None`, so callers can compose subplots.
- **`title=None` means auto-generate** from the column name via `_pretty()` (`"foo_bar"` → `"Foo Bar"`).
- **Validate literal-string arguments explicitly** and `raise ValueError` with a Spanish message — see [fancy_barplot.py:107](utils/fancy_barplot.py#L107).
- **Language split** (stated in [triple_plot.py:3-6](utils/triple_plot.py#L3-L6)): text drawn on the figure and user-facing docstrings in Spanish, code identifiers in English.
- **Reuse rather than re-derive**: `_pretty()`, `_colors()`, `_style()`, `_bar_labels()` in [fancy_barplot.py](utils/fancy_barplot.py); `_fd_bins()` (Freedman–Diaconis bin count) and `_hist_density()` in [triple_plot.py:101](utils/triple_plot.py#L101). Since modules don't cross-import, copy the helper if you need it elsewhere.
