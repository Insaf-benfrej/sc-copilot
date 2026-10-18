import sqlite3

from sc_copilot.data_gen import N_PARTS, N_SUPPLIERS, N_WEEKS, generate


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_tailles_des_tables(tmp_path):
    db = generate(tmp_path / "sc.db")
    with sqlite3.connect(db) as conn:
        assert _count(conn, "suppliers") == N_SUPPLIERS
        assert _count(conn, "parts") == N_PARTS
        assert _count(conn, "demand") == N_PARTS * N_WEEKS
        assert _count(conn, "stock") == N_PARTS
        assert _count(conn, "purchase_orders") == 2500


def test_reproductible(tmp_path):
    """Même graine, mêmes données : indispensable pour des tests fiables."""
    db1 = generate(tmp_path / "a.db", seed=1)
    db2 = generate(tmp_path / "b.db", seed=1)
    query = "SELECT SUM(quantity) FROM demand"
    with sqlite3.connect(db1) as c1, sqlite3.connect(db2) as c2:
        assert c1.execute(query).fetchone() == c2.execute(query).fetchone()


def test_demande_intermittente(tmp_path):
    """Au moins 30 % des semaines doivent avoir une demande nulle."""
    db = generate(tmp_path / "sc.db")
    with sqlite3.connect(db) as conn:
        zeros = conn.execute("SELECT COUNT(*) FROM demand WHERE quantity = 0").fetchone()[0]
        total = _count(conn, "demand")
    assert zeros / total > 0.30


def test_pas_de_valeurs_negatives(tmp_path):
    db = generate(tmp_path / "sc.db")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM demand WHERE quantity < 0").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM stock WHERE quantity < 0").fetchone()[0] == 0
