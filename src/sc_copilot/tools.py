"""Les outils que le LLM peut appeler : la "prise" entre le modèle d'IA et tes calculs.

Le LLM ne calcule jamais. Il lit la liste d'outils ci-dessous (nom, description,
arguments), choisit celui qui convient et demande à ton programme de l'exécuter.
Ce fichier fait trois choses :

1. TOOLS : la liste des outils, décrite pour le LLM (le "menu").
2. call_tool : exécute un outil demandé, après avoir VÉRIFIÉ ses arguments.
   Une erreur ne fait jamais planter le programme : elle est renvoyée au LLM
   sous forme de message, pour qu'il puisse se corriger.
3. run_sql_readonly : permet au LLM de poser une question libre à la base, mais
   en lecture seule (SELECT uniquement, tables autorisées, 200 lignes maximum).

Aucune dépendance au fournisseur : les fonctions to_anthropic_tools() et
to_openai_tools() adaptent le même menu à l'API Claude ou à Azure OpenAI.

Utilisation (démonstration) :
    python -m sc_copilot.tools
"""

import json
import re
import sqlite3
import time
from datetime import date, timedelta

import pandas as pd

from sc_copilot import forecast, inventory, kpi
from sc_copilot.data_gen import N_WEEKS, START_DATE

# Date "d'aujourd'hui" de la base fictive : juste après la dernière semaine de demande.
DATASET_TODAY = (START_DATE + timedelta(weeks=N_WEEKS)).isoformat()

ALLOWED_TABLES = {"suppliers", "parts", "demand", "stock", "purchase_orders"}
MAX_ROWS = 200

SCHEMA = (
    "suppliers(supplier_id, name, reliability); "
    "parts(part_id, category, unit_cost, lead_time_weeks, shelf_life_weeks, hazard_class, supplier_id); "
    "demand(part_id, week_start, quantity); "
    "stock(part_id, quantity, expiry_date); "
    "purchase_orders(order_id, part_id, supplier_id, order_date, promised_date, "
    "received_date, qty_ordered, qty_received)"
)


class ToolError(Exception):
    """Erreur d'utilisation d'un outil (mauvais argument, requête interdite...)."""


# --------------------------------------------------------------------------- le menu des outils
TOOLS = [
    {
        "name": "get_otif",
        "description": (
            "Taux OTIF (On Time In Full) : part des commandes d'achat reçues à l'heure ET "
            "complètes, entre 0 et 1. Global, ou pour un seul fournisseur."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "supplier_id": {
                    "type": "integer",
                    "description": "Numéro du fournisseur (1 à 20). Omettre pour le taux global.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "list_parts_at_risk",
        "description": (
            "Pièces dont la couverture de stock (en semaines) est plus courte que leur délai "
            "d'approvisionnement : une rupture est probable avant la prochaine livraison. "
            "Triées de la plus urgente à la moins urgente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Nombre de pièces à renvoyer (1 à 50, 10 par défaut).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "list_expiring_stock",
        "description": (
            "Stock non nul qui périme bientôt (ou est déjà périmé). days_left négatif = déjà périmé."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Horizon en jours (1 à 365, 30 par défaut)."},
                "limit": {"type": "integer", "description": "Nombre de lignes (1 à 50, 10 par défaut)."},
                "as_of": {
                    "type": "string",
                    "description": f"Date de référence AAAA-MM-JJ. Par défaut : {DATASET_TODAY}.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "forecast_demand",
        "description": (
            "Prévision de la demande hebdomadaire d'une pièce sur les prochaines semaines. "
            "La méthode est choisie automatiquement selon le type de demande."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "part_id": {"type": "integer", "description": "Numéro de la pièce (1 à 300)."},
                "horizon": {"type": "integer", "description": "Nombre de semaines (1 à 26, 4 par défaut)."},
            },
            "required": ["part_id"],
        },
    },
    {
        "name": "recommend_reorder",
        "description": (
            "Recommandation de réapprovisionnement pour une pièce : point de commande, stock de "
            "sécurité, décision (commander ou attendre) et quantité à commander."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "part_id": {"type": "integer", "description": "Numéro de la pièce (1 à 300)."},
                "service_level": {
                    "type": "number",
                    "description": "Niveau de service visé, entre 0,5 et 0,999 (0,95 par défaut).",
                },
                "method": {
                    "type": "string",
                    "enum": ["bootstrap", "normale"],
                    "description": "Méthode de calcul (bootstrap par défaut).",
                },
            },
            "required": ["part_id"],
        },
    },
    {
        "name": "run_sql_readonly",
        "description": (
            "Pose une question libre à la base de données avec une requête SQL SELECT "
            "(lecture seule, 200 lignes maximum). À utiliser seulement si aucun autre outil ne convient. "
            f"Tables : {SCHEMA}. Dates au format AAAA-MM-JJ."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Une requête SELECT unique."}},
            "required": ["query"],
        },
    },
]


def to_anthropic_tools() -> list[dict]:
    """Le menu au format de l'API Claude."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in TOOLS
    ]


def to_openai_tools() -> list[dict]:
    """Le menu au format d'OpenAI et d'Azure OpenAI."""
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
        }
        for t in TOOLS
    ]


# --------------------------------------------------------------------------- vérification des arguments
def _int(args: dict, key: str, default=None, minimum=None, maximum=None, required=False) -> int | None:
    if key not in args or args[key] is None:
        if required:
            raise ToolError(f"Argument obligatoire manquant : {key}.")
        return default
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ToolError(f"{key} doit être un nombre entier.")
    if minimum is not None and value < minimum:
        raise ToolError(f"{key} doit être au moins {minimum}.")
    if maximum is not None and value > maximum:
        raise ToolError(f"{key} doit être au plus {maximum}.")
    return value


def _float(args: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{key} doit être un nombre.")
    if not minimum <= value <= maximum:
        raise ToolError(f"{key} doit être entre {minimum} et {maximum}.")
    return float(value)


def _str(args: dict, key: str, default=None, choices=None, required=False, max_len=1000) -> str | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"Argument obligatoire manquant : {key}.")
        return default
    if not isinstance(value, str):
        raise ToolError(f"{key} doit être du texte.")
    if len(value) > max_len:
        raise ToolError(f"{key} est trop long (maximum {max_len} caractères).")
    if choices is not None and value not in choices:
        raise ToolError(f"{key} doit être l'une de ces valeurs : {', '.join(choices)}.")
    return value


def _check_unknown(args: dict, allowed: set) -> None:
    unknown = set(args) - allowed
    if unknown:
        raise ToolError(f"Argument(s) inconnu(s) : {', '.join(sorted(unknown))}.")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> liste de dictionnaires compatibles JSON (NaN devient null)."""
    return json.loads(df.to_json(orient="records"))


# --------------------------------------------------------------------------- SQL en lecture seule
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


def run_sql_readonly(conn: sqlite3.Connection, query: str, timeout_s: float = 2.0) -> dict:
    """Exécute une requête SELECT unique, avec plusieurs protections."""
    q = query.strip().rstrip(";").strip()
    if not q:
        raise ToolError("Requête vide.")
    if ";" in q:
        raise ToolError("Une seule requête est autorisée.")
    if "--" in q or "/*" in q:
        raise ToolError("Les commentaires SQL sont interdits.")
    if not re.match(r"select\b", q, re.IGNORECASE):
        raise ToolError("Seules les requêtes SELECT sont autorisées.")
    if _FORBIDDEN.search(q):
        raise ToolError("Cette requête contient un mot interdit : seule la lecture est permise.")
    tables = re.findall(r"\b(?:from|join)\s+([A-Za-z_]\w*)", q, re.IGNORECASE)
    unknown = sorted({t for t in tables if t.lower() not in ALLOWED_TABLES})
    if unknown:
        raise ToolError(
            f"Table(s) non autorisée(s) : {', '.join(unknown)}. Tables permises : "
            f"{', '.join(sorted(ALLOWED_TABLES))}."
        )

    deadline = time.monotonic() + timeout_s
    conn.execute("PRAGMA query_only = ON")  # la base refuse toute écriture pendant la requête
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = conn.execute(f"SELECT * FROM ({q}) LIMIT {MAX_ROWS + 1}")
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    except sqlite3.Error as error:
        raise ToolError(f"Erreur SQL : {error}") from error
    finally:
        conn.set_progress_handler(None, 0)
        conn.execute("PRAGMA query_only = OFF")

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    return {
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


# --------------------------------------------------------------------------- les outils eux-mêmes
def _get_otif(conn, args):
    _check_unknown(args, {"supplier_id"})
    supplier_id = _int(args, "supplier_id", minimum=1)
    value = kpi.otif(conn, supplier_id)
    scope = "global" if supplier_id is None else f"fournisseur {supplier_id}"
    return {"scope": scope, "otif": round(value, 4)}


def _list_parts_at_risk(conn, args):
    _check_unknown(args, {"limit"})
    limit = _int(args, "limit", default=10, minimum=1, maximum=50)
    at_risk = kpi.parts_at_risk(conn)
    return {"total_at_risk": len(at_risk), "parts": _records(at_risk.head(limit).round(3))}


def _list_expiring_stock(conn, args):
    _check_unknown(args, {"days", "limit", "as_of"})
    days = _int(args, "days", default=30, minimum=1, maximum=365)
    limit = _int(args, "limit", default=10, minimum=1, maximum=50)
    as_of = _str(args, "as_of", default=DATASET_TODAY, max_len=10)
    try:
        date.fromisoformat(as_of)
    except ValueError as error:
        raise ToolError("as_of doit être une date au format AAAA-MM-JJ.") from error
    expiring = kpi.expiring_stock(conn, as_of=as_of, days=days)
    return {"as_of": as_of, "total": len(expiring), "stock": _records(expiring.head(limit))}


def _forecast_demand(conn, args):
    _check_unknown(args, {"part_id", "horizon"})
    part_id = _int(args, "part_id", required=True, minimum=1)
    horizon = _int(args, "horizon", default=4, minimum=1, maximum=26)
    return {"part_id": part_id, "forecast": _records(forecast.forecast_demand(conn, part_id, horizon).round(3))}


def _recommend_reorder(conn, args):
    _check_unknown(args, {"part_id", "service_level", "method"})
    part_id = _int(args, "part_id", required=True, minimum=1)
    service_level = _float(args, "service_level", default=0.95, minimum=0.5, maximum=0.999)
    method = _str(args, "method", default="bootstrap", choices=("bootstrap", "normale"))
    return inventory.recommend_policy(conn, part_id, service_level=service_level, method=method)


def _run_sql(conn, args):
    _check_unknown(args, {"query"})
    return run_sql_readonly(conn, _str(args, "query", required=True))


HANDLERS = {
    "get_otif": _get_otif,
    "list_parts_at_risk": _list_parts_at_risk,
    "list_expiring_stock": _list_expiring_stock,
    "forecast_demand": _forecast_demand,
    "recommend_reorder": _recommend_reorder,
    "run_sql_readonly": _run_sql,
}


def call_tool(conn: sqlite3.Connection, name: str, arguments=None) -> dict:
    """Exécute un outil demandé par le LLM. Renvoie toujours un dictionnaire JSON.

    Succès : {"ok": True, "result": ...}      Échec : {"ok": False, "error": "..."}
    `arguments` peut être un dictionnaire ou du texte JSON (certains fournisseurs l'envoient ainsi).
    """
    try:
        if name not in HANDLERS:
            raise ToolError(f"Outil inconnu : {name}. Outils disponibles : {', '.join(HANDLERS)}.")
        if arguments is None or arguments == "":
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ToolError("Les arguments ne sont pas du JSON valide.") from error
        if not isinstance(arguments, dict):
            raise ToolError("Les arguments doivent être un objet JSON.")
        return {"ok": True, "result": HANDLERS[name](conn, arguments)}
    except (ToolError, ValueError) as error:  # ValueError : erreurs des fonctions de calcul (pièce inconnue...)
        return {"ok": False, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - on ne veut jamais planter devant le LLM
        return {"ok": False, "error": f"Erreur interne ({type(error).__name__}) : {error}"}


if __name__ == "__main__":
    print("Outils :", ", ".join(t["name"] for t in TOOLS))
    with sqlite3.connect("data/sc.db") as connection:
        for tool, arguments in [
            ("get_otif", {}),
            ("recommend_reorder", {"part_id": 1}),
            ("run_sql_readonly", {"query": "SELECT category, COUNT(*) AS n FROM parts GROUP BY category"}),
            ("run_sql_readonly", {"query": "DELETE FROM parts"}),
        ]:
            print(f"\n{tool} {arguments}")
            print(json.dumps(call_tool(connection, tool, arguments), ensure_ascii=False)[:300])
