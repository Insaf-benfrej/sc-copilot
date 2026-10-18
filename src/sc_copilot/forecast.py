"""Prévision de la demande hebdomadaire par pièce.

Trois idées à retenir :
1. On classe chaque pièce selon deux mesures (classification de Syntetos-Boylan) :
   - ADI : nombre moyen de semaines entre deux demandes (plus c'est grand, plus c'est "intermittent")
   - CV² : variabilité des quantités quand il y a de la demande
2. La méthode adaptée dépend de la classe : lissage exponentiel (SES) pour la demande
   régulière, Croston (variante SBA) pour la demande intermittente.
3. On compare toujours à une base simple (moyenne mobile) : une méthode compliquée
   qui ne bat pas la base ne sert à rien.

Utilisation :
    python -m sc_copilot.forecast
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


# --------------------------------------------------------------------------- classification
def classify_demand(y) -> tuple[float, float, str]:
    """Renvoie (ADI, CV², catégorie) pour une série de demandes hebdomadaires.

    Catégories : smooth (régulière), erratic (régulière mais variable),
    intermittent (rare mais stable), lumpy (rare ET variable), no_demand.
    """
    y = np.asarray(y, dtype=float)
    nonzero = y[y > 0]
    if len(nonzero) == 0:
        return float("inf"), float("nan"), "no_demand"
    adi = len(y) / len(nonzero)
    cv2 = float((nonzero.std() / nonzero.mean()) ** 2) if len(nonzero) > 1 else 0.0
    if adi < ADI_THRESHOLD:
        category = "smooth" if cv2 < CV2_THRESHOLD else "erratic"
    else:
        category = "intermittent" if cv2 < CV2_THRESHOLD else "lumpy"
    return float(adi), cv2, category


# --------------------------------------------------------------------------- méthodes
# Chaque méthode renvoie UNE valeur : la demande moyenne prévue par semaine.
def moving_average(y, window: int = 12) -> float:
    """Base de comparaison : moyenne des `window` dernières semaines."""
    y = np.asarray(y, dtype=float)
    return float(y[-window:].mean())


def ses(y, alpha: float = 0.2) -> float:
    """Lissage exponentiel simple : chaque nouvelle valeur corrige un peu la précédente."""
    y = np.asarray(y, dtype=float)
    level = y[0]
    for value in y[1:]:
        level += alpha * (value - level)
    return float(level)


def croston_sba(y, alpha: float = 0.1) -> float:
    """Croston (variante SBA) pour la demande intermittente.

    On lisse séparément (z) la taille des demandes et (p) l'intervalle entre demandes.
    La prévision par semaine est z / p, corrigée du biais (facteur 1 - alpha/2).
    """
    y = np.asarray(y, dtype=float)
    nonzero = np.flatnonzero(y > 0)
    if len(nonzero) == 0:
        return 0.0
    last = nonzero[0]
    z = y[last]  # taille moyenne d'une demande
    p = last + 1.0  # intervalle moyen entre deux demandes
    for t in range(last + 1, len(y)):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            p += alpha * ((t - last) - p)
            last = t
    return float((1 - alpha / 2) * z / p)


METHODS = {"moyenne_mobile": moving_average, "ses": ses, "croston_sba": croston_sba}


def choose_method(category: str) -> str:
    """Méthode recommandée selon la classe de demande."""
    if category in ("intermittent", "lumpy"):
        return "croston_sba"
    if category in ("smooth", "erratic"):
        return "ses"
    return "moyenne_mobile"


# --------------------------------------------------------------------------- accès aux données
def _load_series(conn: sqlite3.Connection, part_id: int) -> pd.DataFrame:
    series = pd.read_sql(
        "SELECT week_start, quantity FROM demand WHERE part_id = ? ORDER BY week_start",
        conn,
        params=(part_id,),
    )
    if series.empty:
        raise ValueError(f"Aucune demande pour la pièce {part_id}.")
    return series


def forecast_demand(
    conn: sqlite3.Connection, part_id: int, horizon: int = 4, method: str = "auto"
) -> pd.DataFrame:
    """Prévision plate sur `horizon` semaines. Colonnes : week_start, forecast, method, category."""
    series = _load_series(conn, part_id)
    y = series["quantity"].to_numpy()
    _, _, category = classify_demand(y)
    chosen = choose_method(category) if method == "auto" else method
    if chosen not in METHODS:
        raise ValueError(f"Méthode inconnue : {chosen}. Choix : {sorted(METHODS)}")
    value = METHODS[chosen](y)
    last_week = date.fromisoformat(series["week_start"].iloc[-1])
    weeks = [(last_week + timedelta(weeks=h)).isoformat() for h in range(1, horizon + 1)]
    return pd.DataFrame(
        {"week_start": weeks, "forecast": value, "method": chosen, "category": category}
    )


# --------------------------------------------------------------------------- évaluation
def evaluate_methods(conn: sqlite3.Connection, test_weeks: int = 26) -> pd.DataFrame:
    """Compare les méthodes sur les `test_weeks` dernières semaines, par classe de demande.

    Principe : on "cache" les 26 dernières semaines, on prévoit, puis on compare au réel.
    Colonnes du résultat :
    - mae  : erreur absolue moyenne (plus petit = mieux)
    - mase : mae divisée par l'erreur d'une prévision naïve (< 1 = bat la prévision naïve)
    - bias : prévision moyenne - demande réelle moyenne (proche de 0 = sans biais)
    ATTENTION : sur une demande très intermittente, prévoir 0 donne souvent un bon MAE
    mais un mauvais biais. Pour gérer des stocks, regarde les deux.
    """
    demand = pd.read_sql("SELECT part_id, week_start, quantity FROM demand", conn)
    rows = []
    for part_id, group in demand.groupby("part_id"):
        y = group.sort_values("week_start")["quantity"].to_numpy(dtype=float)
        train, test = y[:-test_weeks], y[-test_weeks:]
        _, _, category = classify_demand(train)
        if category == "no_demand":
            continue
        scale = np.abs(np.diff(train)).mean()
        for name, method in METHODS.items():
            forecast = method(train)
            mae = np.abs(test - forecast).mean()
            rows.append(
                {
                    "category": category,
                    "method": name,
                    "mae": mae,
                    "mase": mae / scale if scale > 0 else np.nan,
                    "bias": forecast - test.mean(),
                }
            )
    result = pd.DataFrame(rows)
    return (
        result.groupby(["category", "method"])
        .agg(n_parts=("mae", "size"), mae=("mae", "mean"), mase=("mase", "mean"), bias=("bias", "mean"))
        .round(3)
        .reset_index()
    )


if __name__ == "__main__":
    with sqlite3.connect("data/sc.db") as connection:
        print(forecast_demand(connection, part_id=1).to_string(index=False))
        print()
        print(evaluate_methods(connection).to_string(index=False))
