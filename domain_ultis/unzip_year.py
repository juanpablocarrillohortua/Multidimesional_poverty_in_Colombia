"""Descomprime todos los .zip de una carpeta, agrupados por tipo de archivo.

Uso::

    python domain_ultis/unzip_year.py data/ECV_2025
    python domain_ultis/unzip_year.py data/ECV_2025 --out otra/ruta --force

Para una carpeta ``<origen>`` el resultado queda en
``data/extracted/<origen>_extracted/``, con una subcarpeta por extensión
(``csv/``, ``dta/``, ``sav/``...). Las carpetas internas de cada zip se
descartan: todos los archivos del mismo tipo quedan juntos. Si dos zips
traen un archivo con el mismo nombre, el segundo se guarda como
``<zip>__<nombre>``.

Un archivo que ya existe con el mismo tamaño se omite, así que volver a
ejecutar el script es barato; ``--force`` lo sobrescribe.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
# Ejecutado como archivo, sys.path apunta a domain_ultis/ y no a la raíz.
sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

UTF8_FLAG = 0x800
NO_EXTENSION = "sin_extension"
CHUNK = 8 * 1024 * 1024  # bloques de 8 MB


@dataclass
class Summary:
    extracted: Counter[str] = field(default_factory=Counter)  # por tipo
    skipped: Counter[str] = field(default_factory=Counter)  # por tipo
    renamed: int = 0
    failed: list[Path] = field(default_factory=list)


def _existing_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{raw}' no es una carpeta")
    # resolve() para que `.name` funcione también con "." o "..".
    return path.resolve()


def _member_name(info: zipfile.ZipInfo) -> str:
    """Nombre del archivo dentro del zip, sin carpetas y con tildes bien.

    Si el zip no trae la marca UTF-8, zipfile decodifica el nombre como
    cp437 y "Educación.csv" sale como "Educaci├│n.csv". Cuando recodificarlo
    produce UTF-8 válido, ese era el nombre real.
    """
    name = info.filename
    if not info.flag_bits & UTF8_FLAG:
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    # Quedarse solo con el nombre descarta las carpetas internas y, de paso,
    # cualquier ruta "../" maliciosa.
    return PurePosixPath(name.replace("\\", "/")).name


def _type_dir(name: str) -> str:
    suffix = Path(name).suffix
    return suffix[1:].lower() if suffix else NO_EXTENSION


def _copy_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path
) -> None:
    """Copia por bloques a un .part y lo renombra solo si terminó bien.

    zipfile valida el CRC al final de la lectura: sin el .part, un archivo
    corrupto quedaría escrito con el tamaño correcto y la siguiente
    ejecución lo omitiría.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with zf.open(info) as src, open(partial, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _extract_zip(
    zip_path: Path,
    dest: Path,
    claimed: dict[Path, Path],
    summary: Summary,
    force: bool,
) -> tuple[int, int]:
    """Extrae un zip en dest/<tipo>/. Devuelve (extraídos, omitidos)."""
    extracted = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = _member_name(info)
            if info.is_dir() or name in ("", ".", ".."):
                continue

            kind = _type_dir(name)
            target = dest / kind / name
            # `claimed` recuerda qué zip escribió cada ruta en esta
            # ejecución; como los zips van en orden, los nombres son estables
            # entre ejecuciones.
            if claimed.get(target, zip_path) != zip_path:
                target = target.with_name(f"{zip_path.stem}__{name}")
                summary.renamed += 1
            claimed[target] = zip_path

            if (
                not force
                and target.exists()
                and target.stat().st_size == info.file_size
            ):
                skipped += 1
                summary.skipped[kind] += 1
                continue

            _copy_member(zf, info, target)
            extracted += 1
            summary.extracted[kind] += 1
    return extracted, skipped


def _print_summary(summary: Summary, dest: Path) -> None:
    kinds = sorted(summary.extracted.keys() | summary.skipped.keys())
    print(f"\nResumen ({dest}):")
    if kinds:
        width = max(len(kind) for kind in kinds)
        for kind in kinds:
            print(
                f"  {kind:<{width}}  {summary.extracted[kind]} extraído(s), "
                f"{summary.skipped[kind]} omitido(s)"
            )
    if summary.renamed:
        print(
            f"  {summary.renamed} archivo(s) renombrado(s) por nombre repetido"
        )
    if summary.failed:
        names = ", ".join(path.name for path in summary.failed)
        print(f"  {len(summary.failed)} zip(s) con error: {names}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descomprime todos los .zip de una carpeta en "
            "<out>/<carpeta>_extracted/<tipo de archivo>/."
        ),
    )
    parser.add_argument(
        "source",
        type=_existing_dir,
        help="carpeta que contiene los .zip (p. ej. data/ECV_2025)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.EXTRACTED,
        help="carpeta raíz de destino (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescribe los archivos que ya fueron extraídos",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    zips = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    )
    if not zips:
        print(f"No se encontraron archivos .zip en {args.source}")
        return 1

    dest = args.out / f"{args.source.name}_extracted"
    print(f"Descomprimiendo {len(zips)} zip(s) de {args.source} en {dest}")

    summary = Summary()
    claimed: dict[Path, Path] = {}
    for zip_path in zips:
        try:
            extracted, skipped = _extract_zip(
                zip_path, dest, claimed, summary, args.force
            )
        except (zipfile.BadZipFile, NotImplementedError, OSError) as exc:
            summary.failed.append(zip_path)
            print(f"  [error] {zip_path.name}: {exc}")
            continue
        print(
            f"  [ok] {zip_path.name}: {extracted} extraído(s), "
            f"{skipped} omitido(s)"
        )

    _print_summary(summary, dest)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
