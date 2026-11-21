import sqlite3

import pandas as pd
import pytest

from sc_copilot import analysis, kpi
from sc_copilot.data_gen import generate
from sc_copilot.tools import DATASET_TODAY, call_tool


def make_db(orders, parts=None, demand=None, stock=None):
    """Petite base faite à la main, dont on connaît les bonnes réponses."""
    conn = sqlite3.connect(":memory:")
    parts = parts or [(1, "joint", 10.0, 4, 100, "aucune", 1), (2, "visserie", 5.0, 4, 100, "aucune", 2)]
    pd.DataFrame(parts, columns=["part_id", "category", "unit_cost", "lead_time_weeks", "shelf_life_weeks", "hazard_class", "supplier_id"]).to_sql("parts", conn, index=False)
    pd.DataFrame({"supplier_id": [1, 2], "name": ["A", "B"], "reliability": [0.9, 0.9]}).to_sql("suppliers", conn, index=False)
    columns = ["order_id", "part_id", "supplier_id", "order_date", "promised_date", "received_date", "qty_ordered", "qty_received"]
    pd.DataFrame(orders, columns=columns).to_sql("purchase_orders", conn, index=False)
    weeks = pd.date_range("2025-11-03", periods=8, freq="7D").strftime("%Y-%m-%d")
    demand = demand or [(pid, w, 10) for pid in (1, 2) for w in weeks]
    pd.DataFrame(demand, columns=["part_id", "week_start", "quantity"]).to_sql("demand", conn, index=False)
    stock = stock or [(1, 20, "2027-01-01"), (2, 200, "2027-01-01")]
    pd.DataFrame(stock, columns=["part_id", "quantity", "expiry_date"]).to_sql("stock", conn, index=False)
    return conn


# commandes reçues : fournisseur 1 -> 1 OK, 1 en retard, 1 incomplète ; fournisseur 2 -> 3 OK
RECEIVED = [
    (1, 1, 1, "2025-01-01", "2025-02-01", "2025-02-01", 10, 10),  # OK
    (2, 1, 1, "2025-01-01", "2025-02-01", "2025-02-11", 10, 10),  # 10 jours de retard
    (3, 1, 1, "2025-01-01", "2025-02-01", "2025-02-01", 10, 6),   # incomplète
    (4, 2, 2, "2025-01-01", "2025-02-01", "2025-01-30", 10, 10),  # en avance
    (5, 2, 2, "2025-01-01", "2025-02-01", "2025-02-01", 10, 10),
    (6, 2, 2, "2025-01-01", "2025-02-01", "2025-02-01", 10, 10),
]


# ---- top_deviations
def test_top_deviations_comptes_exacts():
    conn = make_db(RECEIVED)
    table, totals = analysis.top_deviations(conn, "supplier", n=2, min_orders=1)
    assert totals == {"total_orders": 6, "total_problem_orders": 2, "otif": pytest.approx(0.6667, abs=1e-4)}
    first = table.iloc[0]
    assert first["group"] == 1 and first["n_orders"] == 3 and first["problem_orders"] == 2
    assert first["late_orders"] == 1 and first["incomplete_orders"] == 1
    assert first["otif"] == pytest.approx(1 / 3) and first["avg_delay_days_when_late"] == 10
    assert first["share_of_problems"] == 1.0
    assert table.iloc[1]["group"] == 2 and table.iloc[1]["problem_orders"] == 0


def test_top_deviations_otif_global_egal_au_kpi():
    conn = make_db(RECEIVED)
    _, totals = analysis.top_deviations(conn, min_orders=1)
    assert totals["otif"] == pytest.approx(kpi.otif(conn), abs=1e-4)


def test_top_deviations_min_orders_et_rank_by():
    conn = make_db(RECEIVED)
    assert len(analysis.top_deviations(conn, min_orders=4)[0]) == 0
    table, _ = analysis.top_deviations(conn, rank_by="otif", min_orders=1)
    assert table.iloc[0]["group"] == 1


def test_top_deviations_par_categorie_et_piece():
    conn = make_db(RECEIVED)
    by_category, _ = analysis.top_deviations(conn, "category", min_orders=1)
    assert by_category.iloc[0]["group"] == "joint"
    by_part, _ = analysis.top_deviations(conn, "part", min_orders=1)
    assert by_part.iloc[0]["group"] == 1


def test_top_deviations_filtre_de_dates():
    orders = RECEIVED + [(7, 1, 1, "2025-03-01", "2025-04-01", "2025-04-20", 10, 10)]
    conn = make_db(orders)
    table, totals = analysis.top_deviations(conn, since="2025-03-15", min_orders=1)
    assert totals["total_orders"] == 1 and table.iloc[0]["late_orders"] == 1
    assert analysis.top_deviations(conn, until="2025-02-15", min_orders=1)[1]["total_orders"] == 6


def test_top_deviations_arguments_invalides_et_vide():
    conn = make_db(RECEIVED)
    for kwargs in ({"by": "couleur"}, {"rank_by": "x"}, {"n": 0}, {"min_orders": 0}):
        with pytest.raises(ValueError):
            analysis.top_deviations(conn, **kwargs)
    table, totals = analysis.top_deviations(conn, since="2030-01-01")
    assert table.empty and totals["total_orders"] == 0 and totals["otif"] is None


def test_commandes_ouvertes_ignorees_par_otif_et_ecarts():
    orders = RECEIVED + [(8, 1, 1, "2025-12-01", "2025-12-20", None, 10, None)]
    conn = make_db(orders)
    assert kpi.otif(conn) == pytest.approx(4 / 6)
    assert analysis.top_deviations(conn, min_orders=1)[1]["total_orders"] == 6


# ---- delay_risk
# Date de référence 2025-12-29. Pièce 1 : stock 20, demande 10/semaine -> couverture 2 semaines (rupture ~2026-01-12).
def risk_db(open_orders):
    return make_db(RECEIVED + open_orders)


def test_risque_eleve_quand_rupture_avant_arrivee():
    # promise le 2026-02-15 (7 semaines plus tard) : le stock de la pièce 1 est épuisé bien avant
    conn = risk_db([(9, 1, 1, "2025-12-01", "2026-02-15", None, 50, None)])
    row = analysis.delay_risk(conn, "2025-12-29").iloc[0]
    assert row["risk_level"] == "élevé"
    assert row["stockout_date"] == "2026-01-12" and row["coverage_weeks"] == pytest.approx(2.0)
    assert row["days_without_stock"] > 30 and row["overdue_days"] == 0


def test_risque_faible_quand_stock_suffisant_et_fournisseur_fiable():
    # pièce 2 : 200 en stock -> 20 semaines de couverture ; fournisseur 2 jamais en retard
    conn = risk_db([(9, 2, 2, "2025-12-01", "2026-01-15", None, 50, None)])
    row = analysis.delay_risk(conn, "2025-12-29").iloc[0]
    assert row["risk_level"] == "faible" and row["days_without_stock"] == 0
    assert row["supplier_late_rate"] == 0


def test_risque_moyen_commande_deja_en_retard_mais_stock_suffisant():
    conn = risk_db([(9, 2, 2, "2025-10-01", "2025-12-15", None, 50, None)])
    row = analysis.delay_risk(conn, "2025-12-29").iloc[0]
    assert row["overdue_days"] == 14 and row["risk_level"] == "moyen"


def test_fournisseur_peu_fiable_donne_risque_moyen():
    # fournisseur 1 : 1 commande sur 3 en retard (33 % > 30 %) ; stock suffisant (on change la pièce)
    parts = [(1, "joint", 10.0, 4, 100, "aucune", 1), (2, "visserie", 5.0, 4, 100, "aucune", 1)]
    conn = make_db(RECEIVED + [(9, 2, 1, "2025-12-01", "2026-01-15", None, 50, None)], parts=parts)
    row = analysis.delay_risk(conn, "2025-12-29").iloc[0]
    assert row["supplier_late_rate"] == pytest.approx(1 / 3)
    # retard moyen 10 j x 1/3 = 3 j -> arrivée le 2026-01-18 ; couverture de la pièce 2 : 20 semaines
    assert row["expected_arrival"] == "2026-01-18" and row["risk_level"] == "moyen"


def test_ordre_du_classement_et_absence_de_commandes_ouvertes():
    conn = risk_db([
        (9, 2, 2, "2025-12-01", "2026-01-15", None, 50, None),   # faible
        (10, 1, 1, "2025-12-01", "2026-02-15", None, 50, None),  # élevé
    ])
    result = analysis.delay_risk(conn, "2025-12-29")
    assert list(result["order_id"]) == [10, 9]
    assert analysis.delay_risk(make_db(RECEIVED), "2025-12-29").empty
    with pytest.raises(ValueError):
        analysis.delay_risk(conn, "pas une date")


def test_pas_de_demande_pas_de_rupture_prevue():
    demand = [(1, "2025-11-03", 0), (2, "2025-11-03", 10)]
    conn = make_db(RECEIVED + [(9, 1, 1, "2025-12-01", "2026-02-15", None, 50, None)], demand=demand)
    row = analysis.delay_risk(conn, "2025-12-29").iloc[0]
    assert row["days_without_stock"] == 0 and row["risk_level"] != "élevé"


# ---- performance_summary
def test_resume_sur_base_a_la_main():
    conn = risk_db([(9, 1, 1, "2025-12-01", "2026-02-15", None, 50, None)])
    summary = analysis.performance_summary(conn, "2025-03-01", weeks=4)
    # période récente : promises du 2025-02-02 au 2025-03-01 -> aucune commande
    assert summary["otif_recent"] is None and "Aucune commande reçue" in summary["summary_text"]
    summary = analysis.performance_summary(conn, "2025-02-07", weeks=1)
    assert summary["otif_recent"] == pytest.approx(0.6667, abs=1e-4)  # les 6 commandes promises le 2025-02-01
    assert summary["worst_supplier_recent"]["supplier_id"] == 1
    assert "66,7 %" in summary["summary_text"] and "fournisseur 1" in summary["summary_text"]


def test_resume_sur_base_fictive(tmp_path):
    conn = sqlite3.connect(generate(tmp_path / "s.db", open_orders=40))
    summary = analysis.performance_summary(conn, DATASET_TODAY)
    assert summary["open_orders"] == 40 and summary["parts_total"] == 300 and summary["parts_at_risk"] == 152
    assert summary["otif_recent"] == pytest.approx(0.7246, abs=2e-3)
    assert str(summary["parts_at_risk"]) in summary["summary_text"]
    with pytest.raises(ValueError):
        analysis.performance_summary(conn, DATASET_TODAY, weeks=0)


# ---- les nouveaux outils et leurs protections
@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    c = sqlite3.connect(generate(tmp_path_factory.mktemp("db") / "sc.db", open_orders=40))
    yield c
    c.close()


def test_outil_top_deviations(conn):
    res = call_tool(conn, "top_deviations", {})
    assert res["ok"] and len(res["result"]["groups"]) == 3 and res["result"]["by"] == "supplier"
    problems = [g["problem_orders"] for g in res["result"]["groups"]]
    assert problems == sorted(problems, reverse=True)
    assert res["result"]["otif_overall"] == pytest.approx(0.736, abs=1e-3)
    res = call_tool(conn, "top_deviations", {"by": "category", "n": 2, "rank_by": "otif", "since": "2024-01-01"})
    assert res["ok"] and len(res["result"]["groups"]) == 2


def test_outil_delay_risk(conn):
    res = call_tool(conn, "delay_risk", {"limit": 5})
    result = res["result"]
    assert res["ok"] and result["total_open_orders"] == 40 and len(result["orders"]) == 5
    assert sum(result["by_risk_level"].values()) == 40
    high = call_tool(conn, "delay_risk", {"risk_level": "élevé", "limit": 50})["result"]
    assert all(o["risk_level"] == "élevé" for o in high["orders"])
    assert len(high["orders"]) == result["by_risk_level"]["élevé"]


def test_outil_performance_summary(conn):
    res = call_tool(conn, "performance_summary", {"weeks": 8})
    assert res["ok"] and res["result"]["weeks"] == 8 and res["result"]["summary_text"]


@pytest.mark.parametrize(
    "name, args",
    [
        ("top_deviations", {"by": "couleur"}),
        ("top_deviations", {"n": 100}),
        ("top_deviations", {"since": "hier"}),
        ("top_deviations", {"min_orders": 0}),
        ("top_deviations", {"couleur": "rouge"}),
        ("delay_risk", {"risk_level": "énorme"}),
        ("delay_risk", {"as_of": "demain"}),
        ("delay_risk", {"limit": 0}),
        ("performance_summary", {"weeks": 0}),
        ("performance_summary", {"weeks": 100}),
        ("performance_summary", {"as_of": "2025-13-45"}),
    ],
)
def test_nouveaux_outils_refusent_les_mauvais_arguments(conn, name, args):
    res = call_tool(conn, name, args)
    assert res["ok"] is False and res["error"]
