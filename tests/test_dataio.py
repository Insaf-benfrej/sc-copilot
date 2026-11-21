import shutil
import sqlite3

import pandas as pd
import pytest

from sc_copilot import dataio, kpi
from sc_copilot.data_gen import generate
from sc_copilot.dataio import (
    build_db,
    export_csv,
    fill_zero_weeks,
    main,
    validate_folder,
    write_template,
)


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("db") / "sc.db")


@pytest.fixture()
def csv_dir(demo_db, tmp_path):
    folder = tmp_path / "csv"
    export_csv(demo_db, folder)
    return folder


def errors(issues):
    return [i for i in issues if i.severity == "error"]


def edit(folder, table, func):
    path = folder / f"{table}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    result = func(frame)
    (frame if result is None else result).to_csv(path, index=False)


# ---- aller-retour : base fictive -> CSV -> base : mêmes résultats
def test_aller_retour_identique(demo_db, csv_dir, tmp_path):
    frames, issues = validate_folder(csv_dir)
    assert frames is not None and not errors(issues), dataio.format_report(issues)
    new_db = build_db(frames, tmp_path / "new.db")
    with sqlite3.connect(demo_db) as a, sqlite3.connect(new_db) as b:
        assert kpi.otif(a) == kpi.otif(b)
        assert kpi.otif(a, 3) == kpi.otif(b, 3)
        pd.testing.assert_frame_equal(kpi.parts_at_risk(a), kpi.parts_at_risk(b))
        for table in dataio.TABLES:
            left = pd.read_sql(f"SELECT * FROM {table}", a)
            right = pd.read_sql(f"SELECT * FROM {table}", b)
            pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_modele_est_valide(tmp_path):
    write_template(tmp_path / "modele")
    frames, issues = validate_folder(tmp_path / "modele")
    assert frames is not None, dataio.format_report(issues)


# ---- formats de fichiers « à la française »
def test_csv_excel_francais(csv_dir):
    def french(frame):
        frame["reliability"] = frame["reliability"].str.replace(".", ",", regex=False)

    edit(csv_dir, "suppliers", french)
    path = csv_dir / "suppliers.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.to_csv(path, sep=";", index=False, encoding="latin-1")
    edit_dates = pd.read_csv(csv_dir / "purchase_orders.csv", dtype=str, keep_default_na=False)
    for column in ("order_date", "promised_date", "received_date"):
        edit_dates[column] = pd.to_datetime(edit_dates[column]).dt.strftime("%d/%m/%Y")
    edit_dates.to_csv(csv_dir / "purchase_orders.csv", sep=";", index=False)
    frames, issues = validate_folder(csv_dir)
    assert frames is not None, dataio.format_report(issues)
    assert frames["purchase_orders"]["order_date"].iloc[0].count("-") == 2
    assert 0 < frames["suppliers"]["reliability"].iloc[0] < 1


# ---- erreurs repérées, avec un message utile
def test_fichier_absent(csv_dir):
    (csv_dir / "stock.csv").unlink()
    frames, issues = validate_folder(csv_dir)
    assert frames is None and any("stock.csv" in i.message for i in issues)


def test_colonne_obligatoire_absente(csv_dir):
    edit(csv_dir, "parts", lambda f: f.drop(columns=["lead_time_weeks"]))
    frames, issues = validate_folder(csv_dir)
    assert frames is None and any("lead_time_weeks" in i.message for i in errors(issues))


def test_colonne_facultative_absente_et_inconnue(csv_dir):
    edit(csv_dir, "parts", lambda f: f.drop(columns=["hazard_class"]))
    edit(csv_dir, "parts", lambda f: f.insert(0, "Commentaire", "x"))
    frames, issues = validate_folder(csv_dir)
    assert frames is not None
    messages = " ".join(i.message for i in issues)
    assert "hazard_class" in messages and "commentaire" in messages


@pytest.mark.parametrize(
    "table, column, value, expected",
    [
        ("purchase_orders", "promised_date", "pas une date", "promised_date"),
        ("demand", "quantity", "-5", "quantity"),
        ("demand", "quantity", "beaucoup", "quantity"),
        ("suppliers", "reliability", "1.5", "reliability"),
        ("parts", "lead_time_weeks", "0", "lead_time_weeks"),
        ("parts", "part_id", "1.5", "entières"),
        ("purchase_orders", "qty_ordered", "", "vide"),
    ],
)
def test_valeurs_invalides(csv_dir, table, column, value, expected):
    def change(frame):
        frame.loc[2, column] = value

    edit(csv_dir, table, change)
    frames, issues = validate_folder(csv_dir)
    assert frames is None
    assert any(expected in i.message and i.table == table for i in errors(issues)), dataio.format_report(issues)


def test_commande_non_recue_refusee_avec_explication(csv_dir):
    edit(csv_dir, "purchase_orders", lambda f: f.__setitem__("received_date", f["received_date"].where(f.index != 0, "")))
    frames, issues = validate_folder(csv_dir)
    assert frames is None
    assert any("received_date" in i.message and "vide" in i.message for i in errors(issues))


def test_cles_en_double(csv_dir):
    edit(csv_dir, "parts", lambda f: f.__setitem__("part_id", f["part_id"].where(f.index != 1, "1")))
    _, issues = validate_folder(csv_dir)
    assert any("double" in i.message and i.table == "parts" for i in errors(issues))


def test_cle_etrangere_inconnue(csv_dir):
    edit(csv_dir, "stock", lambda f: f.__setitem__("part_id", f["part_id"].where(f.index != 0, "99999")))
    _, issues = validate_folder(csv_dir)
    assert any("99999" in i.message and i.table == "stock" for i in errors(issues))


def test_reception_avant_commande(csv_dir):
    edit(csv_dir, "purchase_orders", lambda f: f.__setitem__("received_date", f["received_date"].where(f.index != 0, "2000-01-01")))
    _, issues = validate_folder(csv_dir)
    assert any("received_date antérieure" in i.message for i in errors(issues))


def test_semaines_non_alignees(csv_dir):
    edit(csv_dir, "demand", lambda f: f.__setitem__("week_start", f["week_start"].where(f.index != 5, "2023-01-04")))
    _, issues = validate_folder(csv_dir)
    assert any("grille" in i.message for i in errors(issues))


# ---- semaines manquantes
def test_semaines_manquantes_avertissement_puis_completees(csv_dir):
    edit(csv_dir, "demand", lambda f: f.drop(index=[3, 4, 5]))
    frames, issues = validate_folder(csv_dir)
    assert frames is not None and any("manquantes" in i.message for i in issues)
    assert len(frames["demand"]) == 300 * 156 - 3

    frames, issues = validate_folder(csv_dir, fill_zero=True)
    assert len(frames["demand"]) == 300 * 156
    assert not any("manquantes" in i.message for i in issues)
    refilled = frames["demand"].set_index(["part_id", "week_start"])["quantity"]
    assert refilled.loc[(1, "2023-01-02")] is not None


def test_fill_zero_weeks_unitaire():
    demand = pd.DataFrame({"part_id": [1, 1, 2], "week_start": ["2025-01-06", "2025-01-20", "2025-01-13"], "quantity": [4, 2, 7]})
    full = fill_zero_weeks(demand, [1, 2])
    assert len(full) == 6
    assert full.loc[(full.part_id == 1) & (full.week_start == "2025-01-13"), "quantity"].iloc[0] == 0
    assert full.loc[(full.part_id == 2) & (full.week_start == "2025-01-13"), "quantity"].iloc[0] == 7


# ---- ligne de commande
def test_cli_template_check_build(tmp_path, capsys):
    folder = str(tmp_path / "m")
    assert main(["template", folder]) == 0
    assert main(["check", folder]) == 0 and "valides" in capsys.readouterr().out
    db = tmp_path / "out" / "sc.db"
    assert main(["build", folder, "--db", str(db)]) == 0 and db.exists()


def test_cli_refuse_les_donnees_invalides(tmp_path, csv_dir, capsys):
    edit(csv_dir, "parts", lambda f: f.drop(columns=["supplier_id"]))
    db = tmp_path / "x.db"
    assert main(["build", str(csv_dir), "--db", str(db)]) == 1
    assert "refusées" in capsys.readouterr().out and not db.exists()


def test_cli_export(demo_db, tmp_path):
    assert main(["export", str(demo_db), str(tmp_path / "e")]) == 0
    assert main(["export", "nulle_part.db", str(tmp_path / "e2")]) == 1
    assert len(list((tmp_path / "e").glob("*.csv"))) == 5


def test_copilote_fonctionne_sur_une_base_chargee(csv_dir, tmp_path):
    """Les outils du copilote marchent sur une base venue de CSV, sans autre adaptation."""
    from sc_copilot.tools import call_tool

    frames, _ = validate_folder(csv_dir)
    db = build_db(frames, tmp_path / "new.db")
    with sqlite3.connect(db) as conn:
        assert call_tool(conn, "get_otif", {})["ok"]
        assert call_tool(conn, "recommend_reorder", {"part_id": 1})["ok"]
        assert call_tool(conn, "forecast_demand", {"part_id": 1})["ok"]
