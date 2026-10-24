import json
import sqlite3

import pytest

from sc_copilot import kpi
from sc_copilot.data_gen import generate
from sc_copilot.tools import (
    DATASET_TODAY,
    MAX_ROWS,
    TOOLS,
    call_tool,
    to_anthropic_tools,
    to_openai_tools,
)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = generate(tmp_path_factory.mktemp("db") / "sc.db")
    c = sqlite3.connect(db)
    yield c
    c.close()


# ---- le menu
def test_menu_bien_forme():
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)) == 6
    for tool in TOOLS:
        assert tool["description"]
        assert tool["parameters"]["type"] == "object"
        for required in tool["parameters"]["required"]:
            assert required in tool["parameters"]["properties"]


def test_formats_fournisseurs():
    anthropic = to_anthropic_tools()
    openai = to_openai_tools()
    assert all({"name", "description", "input_schema"} == set(t) for t in anthropic)
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in openai)
    assert [t["name"] for t in anthropic] == [t["function"]["name"] for t in openai]


def test_date_de_reference():
    assert DATASET_TODAY == "2025-12-29"


# ---- chaque outil renvoie du JSON valide et le même résultat que le calcul direct
def test_get_otif_global_egal_au_calcul_direct(conn):
    res = call_tool(conn, "get_otif", {})
    assert res["ok"] and res["result"]["scope"] == "global"
    assert res["result"]["otif"] == pytest.approx(kpi.otif(conn), abs=1e-4)


def test_get_otif_fournisseur(conn):
    res = call_tool(conn, "get_otif", {"supplier_id": 3})
    assert res["ok"] and res["result"]["scope"] == "fournisseur 3"


def test_pieces_a_risque_limite_et_tri(conn):
    res = call_tool(conn, "list_parts_at_risk", {"limit": 5})
    parts = res["result"]["parts"]
    assert len(parts) == 5
    coverage = [p["coverage_weeks"] for p in parts]
    assert coverage == sorted(coverage)
    assert res["result"]["total_at_risk"] >= 5


def test_stock_qui_expire(conn):
    res = call_tool(conn, "list_expiring_stock", {"days": 60, "limit": 3})
    assert res["ok"] and res["result"]["as_of"] == DATASET_TODAY
    assert len(res["result"]["stock"]) <= 3


def test_prevision(conn):
    res = call_tool(conn, "forecast_demand", {"part_id": 1, "horizon": 6})
    assert res["ok"] and len(res["result"]["forecast"]) == 6


def test_recommandation(conn):
    res = call_tool(conn, "recommend_reorder", {"part_id": 1, "service_level": 0.9, "method": "normale"})
    assert res["ok"]
    assert res["result"]["decision"] in ("commander", "attendre")
    assert res["result"]["method"] == "normale"


def test_toutes_les_sorties_sont_du_json(conn):
    calls = [
        ("get_otif", {}),
        ("list_parts_at_risk", {}),
        ("list_expiring_stock", {}),
        ("forecast_demand", {"part_id": 5}),
        ("recommend_reorder", {"part_id": 5}),
        ("run_sql_readonly", {"query": "SELECT * FROM parts LIMIT 3"}),
    ]
    for name, args in calls:
        res = call_tool(conn, name, args)
        assert res["ok"], (name, res)
        json.dumps(res)  # ne doit pas lever d'erreur


def test_arguments_en_texte_json(conn):
    res = call_tool(conn, "get_otif", '{"supplier_id": 2}')
    assert res["ok"] and res["result"]["scope"] == "fournisseur 2"


# ---- erreurs : jamais d'exception, toujours un message pour le LLM
@pytest.mark.parametrize(
    "name, args",
    [
        ("outil_inconnu", {}),
        ("forecast_demand", {}),                                  # part_id manquant
        ("forecast_demand", {"part_id": "abc"}),                  # mauvais type
        ("forecast_demand", {"part_id": True}),                   # booléen refusé
        ("forecast_demand", {"part_id": 1, "horizon": 500}),      # trop grand
        ("forecast_demand", {"part_id": 99999}),                  # pièce inconnue
        ("forecast_demand", {"part_id": 1, "couleur": "rouge"}),  # argument inconnu
        ("recommend_reorder", {"part_id": 1, "service_level": 1.5}),
        ("recommend_reorder", {"part_id": 1, "method": "magie"}),
        ("list_expiring_stock", {"as_of": "demain"}),
        ("get_otif", {"supplier_id": 9999}),                      # aucune commande
        ("get_otif", "pas du json"),
        ("get_otif", [1, 2]),
    ],
)
def test_erreurs_renvoyees_sans_exception(conn, name, args):
    res = call_tool(conn, name, args)
    assert res["ok"] is False
    assert isinstance(res["error"], str) and res["error"]


# ---- SQL en lecture seule
def test_sql_select_simple(conn):
    res = call_tool(conn, "run_sql_readonly", {"query": "SELECT COUNT(*) AS n FROM parts"})
    assert res["ok"]
    assert res["result"]["columns"] == ["n"] and res["result"]["rows"] == [[300]]


def test_sql_point_virgule_final_accepte(conn):
    assert call_tool(conn, "run_sql_readonly", {"query": "SELECT 1;"})["ok"]


def test_sql_limite_de_lignes(conn):
    res = call_tool(conn, "run_sql_readonly", {"query": "SELECT * FROM demand"})
    assert res["ok"]
    assert res["result"]["row_count"] == MAX_ROWS and res["result"]["truncated"] is True


def test_sql_jointure_autorisee(conn):
    q = "SELECT p.part_id, s.quantity FROM parts p JOIN stock s ON s.part_id = p.part_id LIMIT 2"
    assert call_tool(conn, "run_sql_readonly", {"query": q})["ok"]


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM parts",
        "UPDATE stock SET quantity = 0",
        "DROP TABLE parts",
        "INSERT INTO parts VALUES (1)",
        "SELECT 1; DROP TABLE parts",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM utilisateurs",
        "SELECT 1 -- commentaire",
        "SELECT 1 /* commentaire */",
        "PRAGMA table_info(parts)",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "",
    ],
)
def test_sql_requetes_interdites(conn, query):
    res = call_tool(conn, "run_sql_readonly", {"query": query})
    assert res["ok"] is False


def test_sql_erreur_de_syntaxe_et_base_reste_modifiable(conn):
    res = call_tool(conn, "run_sql_readonly", {"query": "SELECT colonne_inconnue FROM parts"})
    assert res["ok"] is False and "SQL" in res["error"]
    # après une requête refusée, la base doit rester utilisable normalement
    conn.execute("UPDATE stock SET quantity = quantity WHERE part_id = 1")
    assert conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 300
