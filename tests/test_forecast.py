import sqlite3

import numpy as np
import pytest

from sc_copilot.data_gen import generate
from sc_copilot.forecast import (
    classify_demand,
    croston_sba,
    evaluate_methods,
    forecast_demand,
    moving_average,
    ses,
)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = generate(tmp_path_factory.mktemp("db") / "sc.db")
    c = sqlite3.connect(db)
    yield c
    c.close()


# ---- classification : on construit des séries dont on connaît la classe
def test_classe_smooth():
    assert classify_demand([5] * 20)[2] == "smooth"


def test_classe_erratic():
    assert classify_demand([1, 10] * 10)[2] == "erratic"  # toujours de la demande, très variable


def test_classe_intermittent():
    y = ([0, 0, 0, 5] * 5)  # une demande de 5 toutes les 4 semaines
    adi, cv2, category = classify_demand(y)
    assert adi == pytest.approx(4.0)
    assert cv2 == pytest.approx(0.0)
    assert category == "intermittent"


def test_classe_lumpy():
    assert classify_demand([0, 0, 0, 1, 0, 0, 0, 10] * 3)[2] == "lumpy"


def test_classe_sans_demande():
    assert classify_demand([0] * 10)[2] == "no_demand"


# ---- méthodes
def test_moyenne_mobile():
    assert moving_average([1, 2, 3, 4], window=2) == pytest.approx(3.5)


def test_ses_constante():
    assert ses([7] * 30) == pytest.approx(7.0)


def test_croston_demande_reguliere():
    y = [4, 0] * 40  # 4 unités toutes les 2 semaines = 2 par semaine en moyenne
    # SBA applique un facteur 1 - alpha/2 = 0,95 : 0,95 * 4 / 2 = 1,9
    assert croston_sba(y, alpha=0.1) == pytest.approx(1.9, rel=0.02)


def test_croston_sans_demande():
    assert croston_sba([0] * 10) == 0.0


# ---- sur la base fictive
def test_forecast_demand_dates_et_forme(conn):
    fc = forecast_demand(conn, part_id=1, horizon=4)
    assert len(fc) == 4
    last_week = conn.execute("SELECT MAX(week_start) FROM demand").fetchone()[0]
    assert fc["week_start"].iloc[0] > last_week
    assert (fc["forecast"] >= 0).all()


def test_forecast_methode_inconnue(conn):
    with pytest.raises(ValueError):
        forecast_demand(conn, part_id=1, method="magie")


def test_evaluation_a_les_bonnes_colonnes(conn):
    res = evaluate_methods(conn)
    assert {"category", "method", "n_parts", "mae", "mase", "bias"} <= set(res.columns)
    assert set(res["method"]) == {"moyenne_mobile", "ses", "croston_sba"}
    assert np.isfinite(res["mae"]).all()
