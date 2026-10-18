"""Indicateurs (KPI) supply chain calculés à partir de la base SQLite.

Chaque fonction prend une connexion SQLite et renvoie un nombre ou un tableau
pandas. Aucune fonction ne modifie la base.

Utilisation (affiche les KPI de la base fictive) :
    python -m sc_copilot.kpi
"""

import sqlite3

import pandas as pd


def otif(conn: sqlite3.Connection, supplier_id: int | None = None) -> float:
    """OTIF = part des commandes reçues À L'HEURE et COMPLÈTES (entre 0 et 1).

    - à l'heure : date de réception <= date promise
    - complète : quantité reçue >= quantité commandée
    """
    query = "SELECT promised_date, received_date, qty_ordered, qty_received FROM purchase_orders"
    params: tuple = ()
    if supplier_id is not None:
        query += " WHERE supplier_id = ?"
        params = (supplier_id,)
    orders = pd.read_sql(query, conn, params=params)
    if orders.empty:
        raise ValueError("Aucune commande trouvée pour ce filtre.")
    on_time = pd.to_datetime(orders["received_date"]) <= pd.to_datetime(orders["promised_date"])
    complete = orders["qty_received"] >= orders["qty_ordered"]
    return float((on_time & complete).mean())


def coverage_weeks(conn: sqlite3.Connection, window_weeks: int = 12) -> pd.DataFrame:
    """Couverture de stock = stock actuel / demande moyenne par semaine.

    La demande moyenne est calculée sur les `window_weeks` dernières semaines.
    Si la demande moyenne est nulle, la couverture est NaN (infinie en pratique).
    Colonnes : part_id, stock, avg_weekly_demand, lead_time_weeks, coverage_weeks.
    """
    demand = pd.read_sql("SELECT part_id, week_start, quantity FROM demand", conn)
    last_weeks = sorted(demand["week_start"].unique())[-window_weeks:]
    avg_demand = (
        demand[demand["week_start"].isin(last_weeks)]
        .groupby("part_id")["quantity"]
        .mean()
        .rename("avg_weekly_demand")
    )
    stock = pd.read_sql("SELECT part_id, quantity AS stock FROM stock", conn)
    parts = pd.read_sql("SELECT part_id, lead_time_weeks FROM parts", conn)
    result = stock.merge(parts, on="part_id").merge(avg_demand, on="part_id", how="left")
    result["avg_weekly_demand"] = result["avg_weekly_demand"].fillna(0.0)
    safe_demand = result["avg_weekly_demand"].where(result["avg_weekly_demand"] > 0)
    result["coverage_weeks"] = result["stock"] / safe_demand
    return result


def parts_at_risk(conn: sqlite3.Connection, window_weeks: int = 12) -> pd.DataFrame:
    """Pièces dont la couverture est plus courte que le délai d'approvisionnement.

    Autrement dit : on tombera en rupture avant que la prochaine commande arrive.
    Les pièces sans demande récente ne sont pas à risque.
    """
    cov = coverage_weeks(conn, window_weeks)
    at_risk = cov[cov["coverage_weeks"] < cov["lead_time_weeks"]]
    return at_risk.sort_values("coverage_weeks").reset_index(drop=True)


def expiring_stock(conn: sqlite3.Connection, as_of: str, days: int = 30) -> pd.DataFrame:
    """Stock non nul dont la date de péremption tombe dans les `days` jours après `as_of`.

    Inclut aussi le stock déjà périmé (date de péremption avant `as_of`).
    `as_of` est une date au format 'AAAA-MM-JJ'.
    """
    stock = pd.read_sql("SELECT part_id, quantity, expiry_date FROM stock WHERE quantity > 0", conn)
    limit = pd.Timestamp(as_of) + pd.Timedelta(days=days)
    expiry = pd.to_datetime(stock["expiry_date"])
    result = stock[expiry <= limit].copy()
    result["days_left"] = (pd.to_datetime(result["expiry_date"]) - pd.Timestamp(as_of)).dt.days
    return result.sort_values("days_left").reset_index(drop=True)


if __name__ == "__main__":
    with sqlite3.connect("data/sc.db") as connection:
        print(f"OTIF global : {otif(connection):.1%}")
        for sid in (3, 8, 14):
            print(f"OTIF fournisseur {sid} : {otif(connection, sid):.1%}")
        risk = parts_at_risk(connection)
        print(f"Pièces à risque de rupture : {len(risk)} sur {len(coverage_weeks(connection))}")
        print(risk.head(5).to_string(index=False))
