"""Génère une base de données SQLite fictive de type supply chain.

Aucune donnée réelle : tout est tiré au hasard avec une graine fixe,
donc on obtient toujours les mêmes données (utile pour les tests).

Utilisation :
    python -m sc_copilot.data_gen
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

N_SUPPLIERS = 20
N_PARTS = 300
N_WEEKS = 156  # 3 ans de demande hebdomadaire
START_DATE = date(2023, 1, 2)  # un lundi
CATEGORIES = ["joint", "visserie", "adhésif", "peinture", "solvant", "câble"]
HAZARD_CLASSES = ["aucune", "inflammable", "corrosif", "toxique"]


def make_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    """20 fournisseurs, chacun avec une fiabilité entre 0,70 et 0,98."""
    return pd.DataFrame(
        {
            "supplier_id": range(1, N_SUPPLIERS + 1),
            "name": [f"Fournisseur {i:02d}" for i in range(1, N_SUPPLIERS + 1)],
            "reliability": rng.uniform(0.70, 0.98, N_SUPPLIERS).round(3),
        }
    )


def make_parts(rng: np.random.Generator) -> pd.DataFrame:
    """300 pièces avec coût, délai d'approvisionnement et danger chimique."""
    return pd.DataFrame(
        {
            "part_id": range(1, N_PARTS + 1),
            "category": rng.choice(CATEGORIES, N_PARTS),
            "unit_cost": rng.lognormal(mean=3.0, sigma=1.0, size=N_PARTS).round(2),
            "lead_time_weeks": rng.integers(1, 13, N_PARTS),
            "shelf_life_weeks": rng.integers(26, 157, N_PARTS),
            "hazard_class": rng.choice(HAZARD_CLASSES, N_PARTS, p=[0.5, 0.2, 0.15, 0.15]),
            "supplier_id": rng.integers(1, N_SUPPLIERS + 1, N_PARTS),
        }
    )


def make_demand(rng: np.random.Generator, parts: pd.DataFrame) -> pd.DataFrame:
    """Demande hebdomadaire INTERMITTENTE : beaucoup de semaines à zéro.

    Pour chaque pièce, on tire une probabilité p d'avoir de la demande
    une semaine donnée, puis une quantité (loi log-normale) si elle a lieu.
    """
    weeks = [START_DATE + timedelta(weeks=w) for w in range(N_WEEKS)]
    rows = []
    for part_id in parts["part_id"]:
        p = rng.uniform(0.10, 0.90)
        sigma = rng.uniform(0.2, 1.0)  # variabilité des quantités, propre à chaque pièce
        has_demand = rng.random(N_WEEKS) < p
        quantities = np.ceil(rng.lognormal(mean=1.5, sigma=sigma, size=N_WEEKS))
        demand = np.where(has_demand, quantities, 0).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "part_id": part_id,
                    "week_start": [d.isoformat() for d in weeks],
                    "quantity": demand,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_stock(rng: np.random.Generator, parts: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Stock actuel : entre 0 et 12 semaines de demande moyenne, avec une date de péremption."""
    avg_demand = demand.groupby("part_id")["quantity"].mean()
    today = START_DATE + timedelta(weeks=N_WEEKS)
    cover_weeks = rng.uniform(0, 12, N_PARTS)
    quantity = np.round(avg_demand.loc[parts["part_id"]].to_numpy() * cover_weeks).astype(int)
    days_to_expiry = rng.integers(5, 400, N_PARTS)
    return pd.DataFrame(
        {
            "part_id": parts["part_id"],
            "quantity": quantity,
            "expiry_date": [(today + timedelta(days=int(d))).isoformat() for d in days_to_expiry],
        }
    )


def make_purchase_orders(
    rng: np.random.Generator, parts: pd.DataFrame, suppliers: pd.DataFrame
) -> pd.DataFrame:
    """Commandes d'achat, avec 3 ruptures fournisseur injectées (retards de 4 à 8 semaines).

    Une commande est à l'heure avec une probabilité égale à la fiabilité du
    fournisseur. Pendant une rupture, les commandes de ce fournisseur sont
    toutes en retard : les KPI auront donc une histoire à raconter.
    """
    reliability = suppliers.set_index("supplier_id")["reliability"]
    disruptions = {  # fournisseur -> (semaine de début, semaine de fin, retard en semaines)
        3: (40, 52, 6),
        8: (90, 100, 8),
        14: (120, 135, 4),
    }

    rows = []
    n_orders = 2500
    chosen_parts = rng.choice(parts["part_id"].to_numpy(), n_orders)
    for order_id, part_id in enumerate(chosen_parts, start=1):
        part = parts.loc[parts["part_id"] == part_id].iloc[0]
        supplier_id = int(part["supplier_id"])
        order_week = int(rng.integers(0, N_WEEKS - 14))
        order_date = START_DATE + timedelta(weeks=order_week)
        promised = order_date + timedelta(weeks=int(part["lead_time_weeks"]))

        delay_weeks = 0
        if supplier_id in disruptions:
            start, end, delay = disruptions[supplier_id]
            if start <= order_week <= end:
                delay_weeks = delay
        if delay_weeks == 0 and rng.random() > reliability.loc[supplier_id]:
            delay_weeks = int(rng.integers(1, 4))
        received = promised + timedelta(weeks=delay_weeks, days=int(rng.integers(-2, 1)))  # à l'heure ou 1-2 jours en avance

        qty_ordered = int(rng.integers(10, 200))
        short = rng.random() > reliability.loc[supplier_id]  # livraison incomplète
        qty_received = int(qty_ordered * rng.uniform(0.6, 0.95)) if short else qty_ordered

        rows.append(
            {
                "order_id": order_id,
                "part_id": int(part_id),
                "supplier_id": supplier_id,
                "order_date": order_date.isoformat(),
                "promised_date": promised.isoformat(),
                "received_date": received.isoformat(),
                "qty_ordered": qty_ordered,
                "qty_received": qty_received,
            }
        )
    return pd.DataFrame(rows)


def generate(db_path: str | Path = "data/sc.db", seed: int = 42) -> Path:
    """Crée (ou remplace) la base SQLite et renvoie son chemin."""
    rng = np.random.default_rng(seed)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    suppliers = make_suppliers(rng)
    parts = make_parts(rng)
    demand = make_demand(rng, parts)
    stock = make_stock(rng, parts, demand)
    orders = make_purchase_orders(rng, parts, suppliers)

    with sqlite3.connect(db_path) as conn:
        suppliers.to_sql("suppliers", conn, index=False)
        parts.to_sql("parts", conn, index=False)
        demand.to_sql("demand", conn, index=False)
        stock.to_sql("stock", conn, index=False)
        orders.to_sql("purchase_orders", conn, index=False)
    return db_path


if __name__ == "__main__":
    path = generate()
    print(f"Base créée : {path}")
