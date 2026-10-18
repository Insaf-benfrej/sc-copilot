import sqlite3

import numpy as np
import pytest

from sc_copilot.data_gen import generate
from sc_copilot.inventory import (
    evaluate_service_level,
    recommend_policy,
    reorder_point_bootstrap,
    reorder_point_normal,
    safety_stock_normal,
)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = generate(tmp_path_factory.mktemp("db") / "sc.db")
    c = sqlite3.connect(db)
    yield c
    c.close()


# ---- méthode normale : on vérifie la formule avec des chiffres calculés à la main
def test_stock_securite_formule():
    # z(95 %) = 1,6449 ; racine(4 × 2²) = 4 -> 1,6449 × 4 = 6,5795
    assert safety_stock_normal(10, 2, 4, 0.95) == pytest.approx(6.5795, abs=1e-3)


def test_stock_securite_nul_si_demande_constante():
    assert safety_stock_normal(10, 0, 4, 0.95) == 0.0


def test_stock_securite_augmente_avec_le_niveau_de_service():
    assert safety_stock_normal(10, 2, 4, 0.99) > safety_stock_normal(10, 2, 4, 0.90)


def test_stock_securite_augmente_avec_le_delai_incertain():
    assert safety_stock_normal(10, 2, 4, 0.95, lead_time_std=1) > safety_stock_normal(10, 2, 4, 0.95)


def test_point_de_commande():
    # 10 × 4 + 6,5795 = 46,5795
    assert reorder_point_normal(10, 2, 4, 0.95) == pytest.approx(46.5795, abs=1e-3)


def test_niveau_de_service_invalide():
    for bad in (0, 1, -0.5, 1.2):
        with pytest.raises(ValueError):
            safety_stock_normal(10, 2, 4, bad)


# ---- bootstrap
def test_bootstrap_demande_constante():
    # historique : toujours 5 -> sur 4 semaines, toujours 20
    assert reorder_point_bootstrap([5] * 30, 4, 0.95) == 20.0


def test_bootstrap_reproductible():
    history = np.array([0, 0, 3, 0, 8, 0, 0, 2, 0, 5])
    assert reorder_point_bootstrap(history, 3, 0.9, seed=1) == reorder_point_bootstrap(history, 3, 0.9, seed=1)


def test_bootstrap_historique_vide():
    with pytest.raises(ValueError):
        reorder_point_bootstrap([], 3, 0.95)


# ---- recommandation sur la base fictive
def test_recommandation_champs_et_coherence(conn):
    rec = recommend_policy(conn, part_id=1)
    for key in ("reorder_point", "safety_stock", "current_stock", "decision", "order_quantity"):
        assert key in rec
    assert rec["order_quantity"] >= 0
    assert (rec["decision"] == "commander") == (rec["current_stock"] <= rec["reorder_point"])
    if rec["decision"] == "attendre":
        assert rec["order_quantity"] == 0


def test_recommandation_pas_de_commande_si_stock_enorme(conn):
    conn.execute("UPDATE stock SET quantity = 1000000 WHERE part_id = 1")
    assert recommend_policy(conn, part_id=1)["decision"] == "attendre"


def test_recommandation_piece_inconnue(conn):
    with pytest.raises(ValueError):
        recommend_policy(conn, part_id=99999)


def test_recommandation_methode_inconnue(conn):
    with pytest.raises(ValueError):
        recommend_policy(conn, part_id=1, method="magie")


def test_plus_de_service_donne_plus_de_securite(conn):
    low = recommend_policy(conn, part_id=2, service_level=0.80)
    high = recommend_policy(conn, part_id=2, service_level=0.99)
    assert high["reorder_point"] >= low["reorder_point"]


# ---- évaluation
def test_evaluation_colonnes_et_bornes(conn):
    res = evaluate_service_level(conn)
    assert {"category", "method", "n_parts", "achieved", "target"} <= set(res.columns)
    assert set(res["method"]) == {"normale", "bootstrap"}
    assert res["achieved"].between(0, 1).all()
