"""Couche de données générique : lire, vérifier et charger des fichiers CSV dans la base du copilote.

Les outils du copilote ne dépendent pas de la source des données, seulement du format des 5 tables
(suppliers, parts, demand, stock, purchase_orders). Pour utiliser d'autres données (par exemple
des exports d'un système réel), il suffit de fournir 5 fichiers CSV à ce format.

Utilisation :
    python -m sc_copilot.dataio template mes_csv            # écrit des fichiers modèles
    python -m sc_copilot.dataio check mes_csv               # vérifie, sans rien créer
    python -m sc_copilot.dataio build mes_csv --db data/sc.db
    python -m sc_copilot.dataio export data/sc.db csv_demo  # exporte une base existante en CSV
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Col:
    name: str
    kind: str  # "int", "float", "str", "date"
    required: bool = True
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" ou "warning"
    table: str
    message: str

    def __str__(self) -> str:
        return f"[{'ERREUR' if self.severity == 'error' else 'avertissement'}] {self.table} : {self.message}"


SPEC: dict[str, list[Col]] = {
    "suppliers": [
        Col("supplier_id", "int"),
        Col("name", "str", required=False),
        Col("reliability", "float", required=False, min=0, max=1),
    ],
    "parts": [
        Col("part_id", "int"),
        Col("category", "str", required=False),
        Col("unit_cost", "float", required=False, min=0),
        Col("lead_time_weeks", "float", min=0.01),
        Col("shelf_life_weeks", "float", required=False, min=0.01),
        Col("hazard_class", "str", required=False),
        Col("supplier_id", "int"),
    ],
    "demand": [Col("part_id", "int"), Col("week_start", "date"), Col("quantity", "float", min=0)],
    "stock": [Col("part_id", "int"), Col("quantity", "float", min=0), Col("expiry_date", "date", required=False)],
    "purchase_orders": [
        Col("order_id", "int"),
        Col("part_id", "int"),
        Col("supplier_id", "int"),
        Col("order_date", "date"),
        Col("promised_date", "date"),
        Col("received_date", "date"),
        Col("qty_ordered", "float", min=0),
        Col("qty_received", "float", min=0),
    ],
}
TABLES = list(SPEC)
KEYS = {
    "suppliers": ["supplier_id"],
    "parts": ["part_id"],
    "demand": ["part_id", "week_start"],
    "stock": ["part_id"],
    "purchase_orders": ["order_id"],
}
FOREIGN_KEYS = [  # (table, colonne, table référencée, colonne référencée)
    ("parts", "supplier_id", "suppliers", "supplier_id"),
    ("demand", "part_id", "parts", "part_id"),
    ("stock", "part_id", "parts", "part_id"),
    ("purchase_orders", "part_id", "parts", "part_id"),
    ("purchase_orders", "supplier_id", "suppliers", "supplier_id"),
]
EXAMPLES = {
    "suppliers": [[1, "Fournisseur A", 0.92]],
    "parts": [[1, "joint", 12.5, 4, 104, "aucune", 1]],
    "demand": [[1, "2025-01-06", 3], [1, "2025-01-13", 0]],
    "stock": [[1, 20, "2026-06-30"]],
    "purchase_orders": [[1, 1, 1, "2025-01-06", "2025-02-03", "2025-02-05", 50, 50]],
}
MAX_EXAMPLES = 3


# --------------------------------------------------------------------------- lecture
def _read_one(path: Path) -> pd.DataFrame:
    """Lit un CSV en texte : séparateur ',' ou ';' (Excel français), UTF-8 ou Latin-1."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    first_line = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    from io import StringIO

    return pd.read_csv(StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def _rows(index) -> str:
    """Numéros de lignes du fichier (l'en-tête est la ligne 1)."""
    shown = ", ".join(str(int(i) + 2) for i in list(index)[:MAX_EXAMPLES])
    return f"lignes {shown}" + (" ..." if len(index) > MAX_EXAMPLES else "")


def _coerce(table: str, raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[Issue]]:
    issues: list[Issue] = []
    raw = raw.rename(columns=lambda c: str(c).strip().lower())
    columns = {c.name for c in SPEC[table]}
    for extra in sorted(set(raw.columns) - columns):
        issues.append(Issue("warning", table, f"colonne « {extra} » ignorée (inconnue)."))

    out = pd.DataFrame(index=raw.index)
    fatal = False
    for col in SPEC[table]:
        if col.name not in raw.columns:
            if col.required:
                issues.append(Issue("error", table, f"colonne obligatoire absente : {col.name}."))
                fatal = True
            else:
                issues.append(Issue("warning", table, f"colonne facultative absente : {col.name} (laissée vide)."))
                out[col.name] = pd.NA
            continue

        values = raw[col.name].astype(str).str.strip()
        empty = values == ""
        if empty.any() and col.required:
            issues.append(Issue("error", table, f"{col.name} vide ({_rows(raw.index[empty])})."))
            fatal = True
        if col.kind == "str":
            out[col.name] = values.where(~empty, pd.NA)
            continue

        if col.kind in ("int", "float"):
            number = pd.to_numeric(values.str.replace(",", ".", regex=False).where(~empty), errors="coerce")
        else:
            number = None
            iso = pd.to_datetime(values.where(~empty), format="%Y-%m-%d", errors="coerce")
            french = pd.to_datetime(values.where(~empty), format="%d/%m/%Y", errors="coerce")
            date = iso.fillna(french)

        parsed = number if number is not None else date
        bad = parsed.isna() & ~empty
        if bad.any():
            expected = "AAAA-MM-JJ ou JJ/MM/AAAA" if col.kind == "date" else "un nombre"
            issues.append(Issue("error", table, f"{col.name} : valeur non valide, attendu {expected} ({_rows(raw.index[bad])})."))
            fatal = True
            continue

        if col.kind == "date":
            out[col.name] = date.dt.strftime("%Y-%m-%d").where(~empty, pd.NA)
            continue
        if col.kind == "int" and (number.dropna() % 1 != 0).any():
            issues.append(Issue("error", table, f"{col.name} : des valeurs ne sont pas entières."))
            fatal = True
            continue
        if col.min is not None and (number < col.min).any():
            issues.append(Issue("error", table, f"{col.name} : valeur inférieure à {col.min:g} ({_rows(raw.index[number < col.min])})."))
            fatal = True
        if col.max is not None and (number > col.max).any():
            issues.append(Issue("error", table, f"{col.name} : valeur supérieure à {col.max:g} ({_rows(raw.index[number > col.max])})."))
            fatal = True
        out[col.name] = number.astype("Int64") if col.kind == "int" else number

    return (None if fatal else out.reset_index(drop=True)), issues


# --------------------------------------------------------------------------- règles entre tables
def _check_relations(frames: dict[str, pd.DataFrame]) -> list[Issue]:
    issues: list[Issue] = []
    for table, key in KEYS.items():
        dup = frames[table].duplicated(subset=key, keep=False)
        if dup.any():
            issues.append(Issue("error", table, f"clé en double sur ({', '.join(key)}) : {int(dup.sum())} lignes concernées."))

    for table, column, ref_table, ref_column in FOREIGN_KEYS:
        known = set(frames[ref_table][ref_column].dropna())
        missing = ~frames[table][column].isin(known)
        if missing.any():
            sample = ", ".join(str(v) for v in frames[table].loc[missing, column].unique()[:MAX_EXAMPLES])
            issues.append(Issue("error", table, f"{column} absent de {ref_table} pour {int(missing.sum())} lignes (ex. {sample})."))

    orders = frames["purchase_orders"]
    for late, early in (("promised_date", "order_date"), ("received_date", "order_date")):
        wrong = pd.to_datetime(orders[late]) < pd.to_datetime(orders[early])
        if wrong.any():
            issues.append(Issue("error", "purchase_orders", f"{late} antérieure à {early} ({_rows(orders.index[wrong])})."))
    mismatch = orders.merge(frames["parts"][["part_id", "supplier_id"]], on="part_id", suffixes=("", "_part"))
    differing = (mismatch["supplier_id"] != mismatch["supplier_id_part"]).sum()
    if differing:
        issues.append(Issue("warning", "purchase_orders", f"{int(differing)} commandes ont un fournisseur différent de celui de la pièce."))

    demand = frames["demand"]
    if len(demand):
        weeks = pd.to_datetime(demand["week_start"])
        if ((weeks - weeks.min()).dt.days % 7 != 0).any():
            issues.append(Issue("error", "demand", "week_start : les dates ne tombent pas sur une grille de semaines (même jour de la semaine)."))
        else:
            n_weeks = (weeks.max() - weeks.min()).days // 7 + 1
            full = len(frames["parts"]) * n_weeks
            if len(demand) < full:
                issues.append(Issue("warning", "demand", f"semaines manquantes ({len(demand)} lignes sur {full} attendues) : à compléter par 0 avec --fill-zero-weeks si l'absence de ligne veut dire « pas de demande »."))
    if frames["stock"].duplicated("part_id").any():
        pass  # déjà signalé comme clé en double
    return issues


def fill_zero_weeks(demand: pd.DataFrame, part_ids) -> pd.DataFrame:
    """Ajoute une demande de 0 pour chaque pièce et chaque semaine sans ligne."""
    weeks = pd.to_datetime(demand["week_start"])
    grid = pd.date_range(weeks.min(), weeks.max(), freq="7D").strftime("%Y-%m-%d")
    full = pd.MultiIndex.from_product([sorted(part_ids), grid], names=["part_id", "week_start"]).to_frame(index=False)
    merged = full.merge(demand, on=["part_id", "week_start"], how="left")
    merged["quantity"] = merged["quantity"].fillna(0)
    return merged


def validate_folder(folder: str | Path, fill_zero: bool = False) -> tuple[dict[str, pd.DataFrame] | None, list[Issue]]:
    """Lit et vérifie les 5 CSV d'un dossier. Renvoie (tables propres ou None, liste de problèmes)."""
    folder = Path(folder)
    issues: list[Issue] = []
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        path = folder / f"{table}.csv"
        if not path.exists():
            issues.append(Issue("error", table, f"fichier absent : {path.name}."))
            continue
        try:
            raw = _read_one(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            issues.append(Issue("error", table, f"fichier illisible : {error}"))
            continue
        clean, found = _coerce(table, raw)
        issues += found
        if clean is not None:
            frames[table] = clean
    if len(frames) < len(TABLES) or any(i.severity == "error" for i in issues):
        return None, issues

    if fill_zero and len(frames["demand"]):
        frames["demand"] = fill_zero_weeks(frames["demand"], frames["parts"]["part_id"])
    issues += _check_relations(frames)
    if fill_zero:
        issues = [i for i in issues if "semaines manquantes" not in i.message]
    return (None if any(i.severity == "error" for i in issues) else frames), issues


# --------------------------------------------------------------------------- écriture
def build_db(frames: dict[str, pd.DataFrame], db_path: str | Path) -> Path:
    """Crée (ou remplace) la base SQLite à partir de tables déjà vérifiées."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        for table in TABLES:
            frame = frames[table].copy()
            for col in SPEC[table]:
                if col.kind == "float" and col.name in ("quantity", "qty_ordered", "qty_received"):
                    if (frame[col.name].dropna() % 1 == 0).all():
                        frame[col.name] = frame[col.name].astype("Int64")
                if col.kind == "float" and col.name == "lead_time_weeks" and (frame[col.name] % 1 == 0).all():
                    frame[col.name] = frame[col.name].astype("Int64")
            frame = frame.astype(object).where(frame.notna(), None)
            frame.to_sql(table, conn, index=False)
    return db_path


def export_csv(db_path: str | Path, folder: str | Path) -> list[Path]:
    """Exporte une base existante en 5 CSV (utile pour voir le format attendu avec de vraies lignes)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    with sqlite3.connect(db_path) as conn:
        for table in TABLES:
            path = folder / f"{table}.csv"
            pd.read_sql(f"SELECT * FROM {table}", conn).to_csv(path, index=False)
            written.append(path)
    return written


def write_template(folder: str | Path) -> list[Path]:
    """Écrit 5 CSV modèles (en-têtes et une ligne d'exemple) à compléter."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for table in TABLES:
        path = folder / f"{table}.csv"
        pd.DataFrame(EXAMPLES[table], columns=[c.name for c in SPEC[table]]).to_csv(path, index=False)
        written.append(path)
    return written


def format_report(issues: list[Issue]) -> str:
    if not issues:
        return "Aucun problème trouvé."
    return "\n".join(str(i) for i in sorted(issues, key=lambda i: (i.severity != "error", i.table)))


# --------------------------------------------------------------------------- ligne de commande
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vérifier et charger des données CSV pour le copilote.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("check", "vérifie les CSV d'un dossier"), ("build", "vérifie puis crée la base SQLite")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("folder")
        p.add_argument("--fill-zero-weeks", action="store_true", help="semaine sans ligne de demande = demande 0")
        if name == "build":
            p.add_argument("--db", default="data/sc.db")
    p = sub.add_parser("template", help="écrit des CSV modèles")
    p.add_argument("folder")
    p = sub.add_parser("export", help="exporte une base SQLite en CSV")
    p.add_argument("db")
    p.add_argument("folder")
    args = parser.parse_args(argv)

    if args.command == "template":
        for path in write_template(args.folder):
            print(f"Écrit : {path}")
        return 0
    if args.command == "export":
        if not Path(args.db).exists():
            print(f"Base introuvable : {args.db}")
            return 1
        for path in export_csv(args.db, args.folder):
            print(f"Écrit : {path}")
        return 0

    frames, issues = validate_folder(args.folder, fill_zero=args.fill_zero_weeks)
    print(format_report(issues))
    if frames is None:
        print("\nDonnées refusées : corrige les erreurs ci-dessus.")
        return 1
    if args.command == "build":
        print(f"\nBase créée : {build_db(frames, args.db)}")
    else:
        print("\nDonnées valides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
