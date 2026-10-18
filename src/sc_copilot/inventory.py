"""Politique de réapprovisionnement : stock de sécurité et point de commande.

Deux questions auxquelles ce fichier répond, pour chaque pièce :
- À partir de quel niveau de stock faut-il commander ? (le POINT DE COMMANDE)
- Combien de stock en plus faut-il garder "au cas où" ? (le STOCK DE SÉCURITÉ)

Idée : pendant le délai d'approvisionnement (L semaines), la demande est incertaine.
Le point de commande doit couvrir cette demande avec une probabilité choisie,
le NIVEAU DE SERVICE (par exemple 95 % : on accepte une rupture sur 20 cycles).

Deux méthodes :
1. "normale" : formule classique, suppose que la demande suit une loi normale.
2. "bootstrap" : on simule 10 000 délais en retirant au hasard des semaines de
   l'historique réel. Elle ne suppose rien sur la forme de la demande.

Utilisation :
    python -m sc_copilot.inventory
"""

import math
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd

from sc_copilot.forecast import classify_demand


def _check_service_level(service_level: float) -> None:
    if not 0 < service_level < 1:
        raise ValueError("Le niveau de service doit être strictement entre 0 et 1 (ex. 0,95).")


# --------------------------------------------------------------------------- méthode normale
def safety_stock_normal(
    avg_demand: float,
    std_demand: float,
    lead_time_weeks: float,
    service_level: float,
    lead_time_std: float = 0.0,
) -> float:
    """Stock de sécurité = z × racine( L × σ_d² + d² × σ_L² ).

    - z : valeur de la loi normale qui correspond au niveau de service
      (z = 1,645 pour 95 %)
    - L × σ_d² : incertitude due à la demande
    - d² × σ_L² : incertitude due au délai (0 si le délai est certain)
    """
    _check_service_level(service_level)
    z = NormalDist().inv_cdf(service_level)
    variance = lead_time_weeks * std_demand**2 + avg_demand**2 * lead_time_std**2
    return z * math.sqrt(variance)


def reorder_point_normal(
    avg_demand: float,
    std_demand: float,
    lead_time_weeks: float,
    service_level: float,
    lead_time_std: float = 0.0,
) -> float:
    """Point de commande = demande moyenne pendant le délai + stock de sécurité."""
    safety = safety_stock_normal(avg_demand, std_demand, lead_time_weeks, service_level, lead_time_std)
    return avg_demand * lead_time_weeks + safety


# --------------------------------------------------------------------------- méthode bootstrap
def reorder_point_bootstrap(
    history,
    lead_time_weeks: int,
    service_level: float,
    n_sim: int = 10_000,
    seed: int = 0,
) -> float:
    """Point de commande par simulation.

    Pour chaque simulation, on tire `lead_time_weeks` semaines au hasard dans
    l'historique et on additionne leur demande : c'est une demande possible
    pendant un délai. Le point de commande est le quantile au niveau de service
    (95 % des simulations demandent moins que ce nombre).
    """
    _check_service_level(service_level)
    history = np.asarray(history, dtype=float)
    if len(history) == 0:
        raise ValueError("Historique vide.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(history, size=(n_sim, int(lead_time_weeks)), replace=True)
    return float(np.quantile(draws.sum(axis=1), service_level))


# --------------------------------------------------------------------------- recommandation
def recommend_policy(
    conn: sqlite3.Connection,
    part_id: int,
    service_level: float = 0.95,
    window_weeks: int = 52,
    method: str = "bootstrap",
    review_weeks: int = 4,
) -> dict:
    """Recommandation de réapprovisionnement pour une pièce.

    Renvoie un dictionnaire avec le point de commande, le stock de sécurité,
    le stock actuel, la décision ("commander" ou "attendre") et la quantité à commander.
    La quantité est celle qui remonte le stock à : point de commande + demande de
    `review_weeks` semaines.
    """
    _check_service_level(service_level)
    if method not in ("normale", "bootstrap"):
        raise ValueError("method doit être 'normale' ou 'bootstrap'.")

    part = pd.read_sql(
        "SELECT p.lead_time_weeks, COALESCE(s.quantity, 0) AS stock "
        "FROM parts p LEFT JOIN stock s ON s.part_id = p.part_id WHERE p.part_id = ?",
        conn,
        params=(part_id,),
    )
    if part.empty:
        raise ValueError(f"Pièce {part_id} introuvable.")
    lead_time = int(part["lead_time_weeks"].iloc[0])
    stock = int(part["stock"].iloc[0])

    demand = pd.read_sql(
        "SELECT quantity FROM demand WHERE part_id = ? ORDER BY week_start", conn, params=(part_id,)
    )["quantity"].to_numpy(dtype=float)
    if len(demand) == 0:
        raise ValueError(f"Aucune demande pour la pièce {part_id}.")
    recent = demand[-window_weeks:]
    avg, std = float(recent.mean()), float(recent.std(ddof=1)) if len(recent) > 1 else 0.0

    if method == "normale":
        rop = reorder_point_normal(avg, std, lead_time, service_level)
    else:
        rop = reorder_point_bootstrap(recent, lead_time, service_level)
    safety = rop - avg * lead_time

    order_up_to = rop + avg * review_weeks
    quantity = max(0, math.ceil(order_up_to - stock)) if stock <= rop else 0
    return {
        "part_id": part_id,
        "method": method,
        "service_level": service_level,
        "lead_time_weeks": lead_time,
        "avg_weekly_demand": round(avg, 3),
        "reorder_point": round(rop, 1),
        "safety_stock": round(safety, 1),
        "current_stock": stock,
        "decision": "commander" if stock <= rop else "attendre",
        "order_quantity": quantity,
    }


# --------------------------------------------------------------------------- évaluation
def evaluate_service_level(
    conn: sqlite3.Connection, service_level: float = 0.95, test_weeks: int = 26
) -> pd.DataFrame:
    """Le niveau de service visé est-il vraiment atteint ?

    On calcule le point de commande avec les données d'avant les `test_weeks`
    dernières semaines. Puis on regarde, sur ces semaines "cachées", la part des
    délais pendant lesquels la demande réelle est restée sous le point de commande.
    Cette part est le niveau de service RÉEL. Elle doit être proche de la cible.
    Résultat par catégorie de demande et par méthode.
    """
    _check_service_level(service_level)
    parts = pd.read_sql("SELECT part_id, lead_time_weeks FROM parts", conn)
    demand = pd.read_sql("SELECT part_id, week_start, quantity FROM demand", conn)
    rows = []
    for part_id, lead_time in zip(parts["part_id"], parts["lead_time_weeks"]):
        y = demand[demand["part_id"] == part_id].sort_values("week_start")["quantity"].to_numpy(float)
        train, test = y[:-test_weeks], y[-test_weeks:]
        lead_time = int(lead_time)
        _, _, category = classify_demand(train)
        if category == "no_demand" or len(test) < lead_time + 5:
            continue
        windows = np.convolve(test, np.ones(lead_time), mode="valid")  # demande sur chaque délai
        avg, std = train.mean(), train.std(ddof=1)
        rops = {
            "normale": reorder_point_normal(avg, std, lead_time, service_level),
            "bootstrap": reorder_point_bootstrap(train, lead_time, service_level),
        }
        for name, rop in rops.items():
            rows.append(
                {"category": category, "method": name, "achieved": float((windows <= rop).mean())}
            )
    result = pd.DataFrame(rows)
    return (
        result.groupby(["category", "method"])
        .agg(n_parts=("achieved", "size"), achieved=("achieved", "mean"))
        .round(3)
        .reset_index()
        .assign(target=service_level)
    )


if __name__ == "__main__":
    with sqlite3.connect("data/sc.db") as connection:
        for m in ("normale", "bootstrap"):
            print(recommend_policy(connection, part_id=1, method=m))
        print()
        print(evaluate_service_level(connection).to_string(index=False))
