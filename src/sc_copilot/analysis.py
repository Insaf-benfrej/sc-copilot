"""Analyses générales de performance : écarts, risque de retard, résumé.

Ces fonctions ne dépendent que du format des 5 tables (voir dataio.py), pas de la source des données.
Elles ne modifient jamais la base. Chaque règle de décision est écrite en clair dans la docstring.

Utilisation :
    python -m sc_copilot.analysis
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from sc_copilot import kpi

GROUPINGS = {"supplier": "supplier_id", "part": "part_id", "category": "category"}
LATE_RATE_MEDIUM = 0.30  # un fournisseur en retard sur plus de 30 % de ses commandes passées est jugé peu fiable


def _received_orders(conn: sqlite3.Connection, since: str | None = None, until: str | None = None) -> pd.DataFrame:
    """Commandes reçues, avec retard en jours, et indicateurs « en retard » / « incomplète » / « pas OTIF »."""
    query = (
        "SELECT o.order_id, o.part_id, o.supplier_id, o.promised_date, o.received_date, "
        "o.qty_ordered, o.qty_received, p.category "
        "FROM purchase_orders o JOIN parts p ON p.part_id = o.part_id "
        "WHERE o.received_date IS NOT NULL"
    )
    params: list = []
    if since:
        query += " AND o.promised_date >= ?"
        params.append(since)
    if until:
        query += " AND o.promised_date <= ?"
        params.append(until)
    orders = pd.read_sql(query, conn, params=params)
    delay = (pd.to_datetime(orders["received_date"]) - pd.to_datetime(orders["promised_date"])).dt.days
    orders["delay_days"] = delay
    orders["late"] = delay > 0
    orders["incomplete"] = orders["qty_received"] < orders["qty_ordered"]
    orders["problem"] = orders["late"] | orders["incomplete"]  # = pas OTIF
    return orders


def top_deviations(
    conn: sqlite3.Connection,
    by: str = "supplier",
    n: int = 3,
    rank_by: str = "problem_orders",
    since: str | None = None,
    until: str | None = None,
    min_orders: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Les `n` groupes (fournisseurs, pièces ou catégories) qui posent le plus de problèmes de livraison.

    Une commande « pose problème » si elle est en retard OU incomplète (donc pas OTIF).
    - rank_by="problem_orders" : le plus grand nombre de commandes à problème d'abord (impact).
    - rank_by="otif"           : le taux OTIF le plus bas d'abord (gravité).
    Seuls les groupes d'au moins `min_orders` commandes sont classés (éviter les petits échantillons).
    `since` / `until` : bornes (incluses) sur la date promise, au format AAAA-MM-JJ.

    Renvoie (tableau, totaux). Colonnes : group, n_orders, otif, problem_orders, late_orders,
    incomplete_orders, avg_delay_days_when_late, share_of_problems.
    """
    if by not in GROUPINGS:
        raise ValueError(f"by doit être l'une de ces valeurs : {', '.join(GROUPINGS)}.")
    if rank_by not in ("problem_orders", "otif"):
        raise ValueError("rank_by doit être 'problem_orders' ou 'otif'.")
    if n < 1 or min_orders < 1:
        raise ValueError("n et min_orders doivent être au moins 1.")

    orders = _received_orders(conn, since, until)
    total_problems = int(orders["problem"].sum())
    totals = {
        "total_orders": len(orders),
        "total_problem_orders": total_problems,
        "otif": round(1 - total_problems / len(orders), 4) if len(orders) else None,
    }
    if orders.empty:
        return pd.DataFrame(columns=["group", "n_orders", "otif", "problem_orders", "late_orders",
                                     "incomplete_orders", "avg_delay_days_when_late", "share_of_problems"]), totals

    orders["delay_when_late"] = orders["delay_days"].where(orders["late"])
    grouped = orders.groupby(GROUPINGS[by]).agg(
        n_orders=("order_id", "size"),
        problem_orders=("problem", "sum"),
        late_orders=("late", "sum"),
        incomplete_orders=("incomplete", "sum"),
        avg_delay_days_when_late=("delay_when_late", "mean"),
    )
    grouped = grouped[grouped["n_orders"] >= min_orders].copy()
    grouped["otif"] = 1 - grouped["problem_orders"] / grouped["n_orders"]
    grouped["share_of_problems"] = grouped["problem_orders"] / total_problems if total_problems else 0.0
    grouped = grouped.reset_index().rename(columns={GROUPINGS[by]: "group"})
    if rank_by == "problem_orders":
        grouped = grouped.sort_values(["problem_orders", "otif", "group"], ascending=[False, True, True])
    else:
        grouped = grouped.sort_values(["otif", "problem_orders", "group"], ascending=[True, False, True])
    columns = ["group", "n_orders", "otif", "problem_orders", "late_orders", "incomplete_orders",
               "avg_delay_days_when_late", "share_of_problems"]
    return grouped[columns].head(n).reset_index(drop=True), totals


def delay_risk(conn: sqlite3.Connection, as_of: str) -> pd.DataFrame:
    """Commandes encore ouvertes (pas de date de réception) classées par risque de retard.

    Pour chaque commande ouverte :
    - retard actuel : jours écoulés depuis la date promise (si `as_of` est après).
    - fiabilité du fournisseur : part de ses commandes passées reçues en retard, et retard moyen quand
      il est en retard (à défaut d'historique : moyennes de tous les fournisseurs).
    - arrivée attendue = date promise + (part en retard x retard moyen) ; jamais avant `as_of`.
    - rupture prévue = `as_of` + couverture de stock actuelle (semaines). Le stock en commande n'est PAS compté.
    Niveau de risque :
    - « élevé »  : la rupture est prévue avant l'arrivée attendue ;
    - « moyen »  : pas de rupture prévue, mais la commande est déjà en retard ou le fournisseur est en retard
                   sur plus de 30 % de ses commandes ;
    - « faible » : sinon.
    """
    pd.Timestamp(as_of)  # lève ValueError si la date est invalide
    today = pd.Timestamp(as_of)
    open_orders = pd.read_sql(
        "SELECT order_id, part_id, supplier_id, promised_date, qty_ordered "
        "FROM purchase_orders WHERE received_date IS NULL",
        conn,
    )
    columns = ["order_id", "part_id", "supplier_id", "promised_date", "qty_ordered", "overdue_days",
               "supplier_late_rate", "supplier_avg_delay_days_when_late", "expected_arrival",
               "coverage_weeks", "stockout_date", "days_without_stock", "risk_level"]
    if open_orders.empty:
        return pd.DataFrame(columns=columns)

    history = _received_orders(conn)
    history["delay_when_late"] = history["delay_days"].where(history["late"])
    overall_rate = float(history["late"].mean()) if len(history) else 0.0
    overall_delay = float(history["delay_when_late"].mean()) if history["late"].any() else 0.0
    per_supplier = history.groupby("supplier_id").agg(
        supplier_late_rate=("late", "mean"), supplier_avg_delay_days_when_late=("delay_when_late", "mean")
    )
    result = open_orders.merge(per_supplier, on="supplier_id", how="left")
    result["supplier_late_rate"] = result["supplier_late_rate"].fillna(overall_rate)
    result["supplier_avg_delay_days_when_late"] = result["supplier_avg_delay_days_when_late"].fillna(overall_delay)

    promised = pd.to_datetime(result["promised_date"])
    result["overdue_days"] = (today - promised).dt.days.clip(lower=0)
    expected_delay = result["supplier_late_rate"] * result["supplier_avg_delay_days_when_late"].fillna(0.0)
    arrival = promised + pd.to_timedelta(expected_delay.round().astype(int), unit="D")
    arrival = arrival.where(arrival >= today, today)
    result["expected_arrival"] = arrival.dt.strftime("%Y-%m-%d")

    cover = kpi.coverage_weeks(conn)[["part_id", "coverage_weeks"]]
    result = result.merge(cover, on="part_id", how="left")
    stockout = today + pd.to_timedelta((result["coverage_weeks"] * 7).round(), unit="D")
    result["stockout_date"] = stockout.dt.strftime("%Y-%m-%d")
    gap = (arrival - stockout).dt.days
    result["days_without_stock"] = gap.where(gap > 0, 0).fillna(0).astype(int)

    stockout_first = result["coverage_weeks"].notna() & (stockout < arrival)
    unreliable = (result["overdue_days"] > 0) | (result["supplier_late_rate"] > LATE_RATE_MEDIUM)
    result["risk_level"] = "faible"
    result.loc[unreliable & ~stockout_first, "risk_level"] = "moyen"
    result.loc[stockout_first, "risk_level"] = "élevé"

    rank = result["risk_level"].map({"élevé": 0, "moyen": 1, "faible": 2})
    result = result.assign(_rank=rank).sort_values(
        ["_rank", "days_without_stock", "overdue_days", "order_id"], ascending=[True, False, False, True]
    )
    return result[columns].reset_index(drop=True)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + " %"


def performance_summary(conn: sqlite3.Connection, as_of: str, weeks: int = 12) -> dict:
    """Résumé de performance : OTIF récent vs période précédente, pièces à risque, commandes ouvertes, principal écart.

    Période récente = commandes dont la date promise tombe dans les `weeks` semaines avant `as_of`
    (comparée aux `weeks` semaines d'avant). Le texte `summary_text` est écrit par le programme, pas par un LLM.
    """
    if weeks < 1:
        raise ValueError("weeks doit être au moins 1.")
    today = pd.Timestamp(as_of)
    start = today - pd.Timedelta(weeks=weeks)
    previous_start = start - pd.Timedelta(weeks=weeks)
    fmt = "%Y-%m-%d"

    recent, recent_totals = top_deviations(
        conn, by="supplier", n=1, since=(start + pd.Timedelta(days=1)).strftime(fmt), until=today.strftime(fmt), min_orders=3
    )
    _, previous_totals = top_deviations(
        conn, by="supplier", n=1, since=(previous_start + pd.Timedelta(days=1)).strftime(fmt), until=start.strftime(fmt), min_orders=3
    )
    otif_now, otif_before = recent_totals["otif"], previous_totals["otif"]
    delta_points = round((otif_now - otif_before) * 100, 1) if otif_now is not None and otif_before is not None else None

    cov = kpi.coverage_weeks(conn)
    at_risk = int((cov["coverage_weeks"] < cov["lead_time_weeks"]).sum())
    open_orders = delay_risk(conn, as_of)
    overdue = int((open_orders["overdue_days"] > 0).sum()) if len(open_orders) else 0
    high = int((open_orders["risk_level"] == "élevé").sum()) if len(open_orders) else 0
    worst = None
    if len(recent) and int(recent.loc[0, "problem_orders"]) > 0:
        row = recent.iloc[0]
        worst = {"supplier_id": int(row["group"]), "problem_orders": int(row["problem_orders"]),
                 "n_orders": int(row["n_orders"]), "otif": round(float(row["otif"]), 4)}

    lines = []
    if otif_now is None:
        lines.append(f"Aucune commande reçue dont la date promise tombe dans les {weeks} dernières semaines avant {as_of}.")
    else:
        line = f"OTIF sur les {weeks} dernières semaines avant {as_of} : {_pct(otif_now)} ({recent_totals['total_orders']} commandes)"
        if otif_before is not None:
            line += f", contre {_pct(otif_before)} sur la période précédente"
            line += f" ({delta_points:+.1f} points)".replace(".", ",") if delta_points is not None else ""
        lines.append(line + ".")
    lines.append(f"{at_risk} pièces sur {len(cov)} sont à risque de rupture (couverture inférieure au délai d'approvisionnement).")
    lines.append(f"{len(open_orders)} commandes ouvertes, dont {overdue} en retard et {high} à risque élevé de rupture avant livraison.")
    if worst:
        lines.append(
            f"Principal écart récent : fournisseur {worst['supplier_id']} avec {worst['problem_orders']} commandes sur "
            f"{worst['n_orders']} en retard ou incomplètes."
        )
    return {
        "as_of": as_of,
        "weeks": weeks,
        "otif_recent": otif_now,
        "otif_previous": otif_before,
        "otif_change_points": delta_points,
        "orders_recent": recent_totals["total_orders"],
        "parts_at_risk": at_risk,
        "parts_total": len(cov),
        "open_orders": len(open_orders),
        "open_orders_overdue": overdue,
        "open_orders_high_risk": high,
        "worst_supplier_recent": worst,
        "summary_text": " ".join(lines),
    }


if __name__ == "__main__":
    from sc_copilot.tools import DATASET_TODAY

    with sqlite3.connect("data/sc.db") as connection:
        table, totals = top_deviations(connection)
        print("Top 3 fournisseurs (nombre de commandes à problème) :")
        print(table.round(3).to_string(index=False))
        print()
        print(performance_summary(connection, DATASET_TODAY)["summary_text"])
