"""Tests des KPI sur un mini-jeu de données fait à la main, dont on connaît les réponses."""

import sqlite3

import pandas as pd
import pytest

from sc_copilot.kpi import coverage_weeks, expiring_stock, otif, parts_at_risk


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")

    # 5 commandes : (à l'heure, complète) = (oui,oui) (oui,non) (non,oui) (oui,oui) (non,non)
    pd.DataFrame(
        {
            "supplier_id": [1, 1, 1, 2, 2],
            "promised_date": ["2025-01-10"] * 5,
            "received_date": ["2025-01-10", "2025-01-09", "2025-01-12", "2025-01-08", "2025-01-15"],
            "qty_ordered": [10, 10, 10, 10, 10],
            "qty_received": [10, 6, 10, 10, 5],
        }
    ).to_sql("purchase_orders", c, index=False)

    # 3 pièces : 1 = à risque, 2 = couverte, 3 = sans demande
    pd.DataFrame({"part_id": [1, 2, 3], "lead_time_weeks": [6, 3, 4]}).to_sql(
        "parts", c, index=False
    )
    pd.DataFrame({"part_id": [1, 2, 3], "quantity": [20, 100, 10],
                  "expiry_date": ["2025-01-15", "2025-02-15", "2024-12-20"]}
                 ).to_sql("stock", c, index=False)
    weeks = ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"]
    rows = []
    for w in weeks:
        rows += [(1, w, 5), (2, w, 10), (3, w, 0)]
    pd.DataFrame(rows, columns=["part_id", "week_start", "quantity"]).to_sql(
        "demand", c, index=False
    )
    yield c
    c.close()


def test_otif_global(conn):
    # 2 commandes sur 5 sont à l'heure ET complètes
    assert otif(conn) == pytest.approx(0.4)


def test_otif_par_fournisseur(conn):
    assert otif(conn, supplier_id=1) == pytest.approx(1 / 3)
    assert otif(conn, supplier_id=2) == pytest.approx(0.5)


def test_otif_sans_commande(conn):
    with pytest.raises(ValueError):
        otif(conn, supplier_id=99)


def test_couverture(conn):
    cov = coverage_weeks(conn, window_weeks=4).set_index("part_id")
    assert cov.loc[1, "coverage_weeks"] == pytest.approx(4.0)   # 20 / 5
    assert cov.loc[2, "coverage_weeks"] == pytest.approx(10.0)  # 100 / 10
    assert pd.isna(cov.loc[3, "coverage_weeks"])                # demande nulle


def test_pieces_a_risque(conn):
    risk = parts_at_risk(conn, window_weeks=4)
    # pièce 1 : couverture 4 < délai 6 -> à risque ; 2 et 3 non
    assert list(risk["part_id"]) == [1]


def test_stock_qui_expire(conn):
    exp = expiring_stock(conn, as_of="2025-01-01", days=30)
    # pièce 3 déjà périmée, pièce 1 expire dans 14 jours, pièce 2 est trop loin
    assert list(exp["part_id"]) == [3, 1]
    assert list(exp["days_left"]) == [-12, 14]
