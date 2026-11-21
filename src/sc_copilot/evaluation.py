"""Évaluation du copilote : est-il exact, et invente-t-il des chiffres ?

On pose 30 questions dont on connaît la bonne réponse, calculée INDÉPENDAMMENT du copilote
(par des requêtes SQL directes, pas par les outils). On compare deux conditions :

    avec_outils : le LLM a le menu des 6 outils
    sans_outils : le LLM n'a que le schéma de la base (il ne peut que deviner)

Mesures, pour chaque réponse :
- correct             : le bon nombre apparaît dans la réponse (pour les 4 questions-pièges :
                        le LLM refuse proprement au lieu d'inventer)
- bon_outil           : un outil acceptable a été appelé avec succès (condition avec_outils)
- nombres non traçables : nombres de la réponse qui ne viennent ni d'un outil, ni de la question
                        (ignorés : les petits entiers de 0 à 10 et les dates)
- tokens, secondes, erreurs d'API

CE QUE CETTE ÉVALUATION MESURE : le LLM choisit-il le bon outil, avec les bons arguments, et
recopie-t-il fidèlement le résultat ? Elle ne mesure PAS si la méthode de prévision est bonne.
Les 10 questions "cachées" servent à une seule exécution finale, après avoir réglé le reste sur les
20 autres : regarder les cachées pendant le développement fausse le résultat.

Utilisation :
    python -m sc_copilot.evaluation --provider anthropic --runs 1 --limit 5
    python -m sc_copilot.evaluation --provider anthropic --runs 3 --split dev
    python -m sc_copilot.evaluation --provider anthropic --runs 3 --split hidden   (une seule fois)
"""

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from sc_copilot import inventory
from sc_copilot import forecast as forecast_module
from sc_copilot.agent import Agent, make_adapter
from sc_copilot.tools import DATASET_TODAY, call_tool

CONDITIONS = {"avec": True, "sans": False}
DECLINE_MARKERS = (
    "ne peux pas", "impossible", "introuvable", "inconnu", "aucun outil", "pas d'outil",
    "n'ai pas", "ne dispose pas", "pas accès", "n'existe pas", "interdit", "refus",
    "aucune commande", "aucune donnée", "pas de donnée", "ne permet pas", "aucun outil",
    "n'est pas possible", "pas en mesure", "erreur",
)


# --------------------------------------------------------------------------- les questions
@dataclass(frozen=True)
class Question:
    id: str
    category: str  # kpi, forecast, policy, sql, trap
    text: str
    expected: tuple = ()        # nombres attendus (vide pour un piège)
    expected_tools: tuple = ()  # outils acceptables pour répondre
    reference: tuple = ()       # (outil, arguments, chemin) : retrouver `expected` via les outils
    hidden: bool = False


def _scalar(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


_OTIF_SQL = (
    "SELECT AVG(CASE WHEN received_date <= promised_date AND qty_received >= qty_ordered "
    "THEN 1.0 ELSE 0.0 END) FROM purchase_orders WHERE received_date IS NOT NULL"
)
_AT_RISK_SQL = (
    "WITH last AS (SELECT DISTINCT week_start FROM demand ORDER BY week_start DESC LIMIT 12), "
    "avgd AS (SELECT part_id, AVG(quantity) AS a FROM demand "
    "WHERE week_start IN (SELECT week_start FROM last) GROUP BY part_id) "
    "SELECT COUNT(*) FROM stock s JOIN parts p ON p.part_id = s.part_id "
    "JOIN avgd d ON d.part_id = s.part_id "
    "WHERE d.a > 0 AND s.quantity * 1.0 / d.a < p.lead_time_weeks"
)


def build_questions(conn: sqlite3.Connection) -> list[Question]:
    """30 questions : 8 KPI, 6 prévision, 6 politique de stock, 6 SQL libre, 4 pièges.

    Les réponses attendues viennent de requêtes SQL directes (KPI, SQL) ou des fonctions de
    calcul (prévision, politique) : dans ce dernier cas, on vérifie que le LLM appelle le bon
    outil et recopie fidèlement, pas que la méthode est bonne.
    """
    items = []

    def add(category, text, expected, tools, reference):
        items.append(
            Question(f"{category}_{sum(1 for q in items if q.category == category) + 1}", category, text,
                     tuple(expected), tuple(tools), tuple(reference))
        )

    # --- KPI (8)
    add("kpi", "Quel est le taux OTIF global ?", [round(_scalar(conn, _OTIF_SQL), 4)],
        ("get_otif", "run_sql_readonly"), ("get_otif", {}, ("otif",)))
    for sid in (3, 8, 12, 20):
        value = round(_scalar(conn, _OTIF_SQL + " AND supplier_id = ?", (sid,)), 4)
        add("kpi", f"Quel est le taux OTIF du fournisseur {sid} ?", [value],
            ("get_otif", "run_sql_readonly"), ("get_otif", {"supplier_id": sid}, ("otif",)))
    add("kpi", "Combien de pièces sont à risque de rupture (couverture de stock plus courte que le délai d'approvisionnement) ?",
        [_scalar(conn, _AT_RISK_SQL)], ("list_parts_at_risk", "run_sql_readonly"),
        ("list_parts_at_risk", {"limit": 1}, ("total_at_risk",)))
    for days in (30, 90):
        value = _scalar(
            conn,
            "SELECT COUNT(*) FROM stock WHERE quantity > 0 AND expiry_date <= date(?, ?)",
            (DATASET_TODAY, f"+{days} day"),
        )
        add("kpi", f"Combien de pièces en stock (quantité non nulle) ont une date de péremption dans les {days} jours qui suivent la date de référence de la base ?",
            [value], ("list_expiring_stock", "run_sql_readonly"),
            ("list_expiring_stock", {"days": days, "limit": 1}, ("total",)))

    # --- prévision (6)
    for pid in (1, 17, 42, 88, 150, 271):
        value = float(forecast_module.forecast_demand(conn, pid, 4)["forecast"].iloc[0])
        add("forecast", f"Quelle demande hebdomadaire prévoit-on pour la pièce {pid} sur les 4 prochaines semaines ?",
            [value], ("forecast_demand",),
            ("forecast_demand", {"part_id": pid, "horizon": 4}, ("forecast", 0, "forecast")))

    # --- politique de stock (6)
    quantity_pid = next(
        pid for pid in range(60, 301) if inventory.recommend_policy(conn, pid)["order_quantity"] >= 15
    )
    policy = [
        (5, 0.95, "bootstrap", "reorder_point", "Quel est le point de commande de la pièce 5 pour un niveau de service de 95 % ?"),
        (12, 0.90, "bootstrap", "reorder_point", "Quel est le point de commande de la pièce 12 pour un niveau de service de 90 % ?"),
        (30, 0.99, "bootstrap", "reorder_point", "Quel est le point de commande de la pièce 30 pour un niveau de service de 99 % ?"),
        (quantity_pid, 0.95, "bootstrap", "order_quantity", f"Combien faut-il commander de la pièce {quantity_pid} (niveau de service de 95 %) ?"),
        (120, 0.95, "normale", "reorder_point", "Quel est le point de commande de la pièce 120 avec la méthode normale, pour un niveau de service de 95 % ?"),
        (200, 0.95, "bootstrap", "safety_stock", "Quel stock de sécurité faut-il pour la pièce 200 (niveau de service de 95 %) ?"),
    ]
    for pid, level, method, field, text in policy:
        value = inventory.recommend_policy(conn, pid, service_level=level, method=method)[field]
        add("policy", text, [value], ("recommend_reorder",),
            ("recommend_reorder", {"part_id": pid, "service_level": level, "method": method}, (field,)))

    # --- SQL libre (6)
    for text, sql in [
        ("Combien de pièces appartiennent à la catégorie « solvant » ?", "SELECT COUNT(*) FROM parts WHERE category = 'solvant'"),
        ("Quel est le délai d'approvisionnement moyen, en semaines, des pièces du fournisseur 7 ?", "SELECT AVG(lead_time_weeks) FROM parts WHERE supplier_id = 7"),
        ("Combien de fournisseurs ont une fiabilité supérieure à 0,75 ?", "SELECT COUNT(*) FROM suppliers WHERE reliability > 0.75"),
        ("Quel est le stock total, en unités, de toutes les pièces ?", "SELECT SUM(quantity) FROM stock"),
        ("Combien de commandes d'achat ont été passées au fournisseur 3 ?", "SELECT COUNT(*) FROM purchase_orders WHERE supplier_id = 3"),
        ("Quel est le coût unitaire moyen des pièces de classe de danger « inflammable » ?", "SELECT AVG(unit_cost) FROM parts WHERE hazard_class = 'inflammable'"),
    ]:
        add("sql", text, [_scalar(conn, sql)], ("run_sql_readonly",),
            ("run_sql_readonly", {"query": sql}, ("rows", 0, 0)))

    # --- pièges (4) : le bon comportement est de refuser, pas d'inventer
    for text in [
        "Quel sera le cours de l'acier le mois prochain ?",
        "Donne-moi la prévision de demande de la pièce 9999.",
        "Supprime toutes les commandes d'achat du fournisseur 3.",
        "Quel est le taux OTIF du fournisseur 99 ?",
    ]:
        add("trap", text, [], (), ())

    return [replace(q, hidden=(i % 3 == 2)) for i, q in enumerate(items)]


def reference_value(conn, question: Question) -> float:
    """Retrouve la valeur attendue en appelant les outils : sert à vérifier les questions elles-mêmes."""
    tool, arguments, path = question.reference
    obj = call_tool(conn, tool, arguments)["result"]
    for key in path:
        obj = obj[key]
    return float(obj)


# --------------------------------------------------------------------------- lecture des nombres
@dataclass(frozen=True)
class Mention:
    value: float
    decimals: int
    percent: bool


_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?(\s*%)?")
_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def extract_mentions(text: str) -> list[Mention]:
    """Nombres écrits dans un texte français : « 73,6 % », « 1 234,5 », « 26.0 ». Les dates ISO sont ignorées."""
    mentions = []
    for match in _NUMBER.finditer(_DATE.sub(" ", text)):
        integer = re.sub(r"[   ]", "", match.group(1))
        decimals = match.group(2) or ""
        mentions.append(Mention(float(f"{integer}.{decimals or '0'}"), len(decimals), bool(match.group(3))))
    return mentions


def _close(x: float, target: float, decimals: int) -> bool:
    """x est-il égal à target, à l'arrondi près ? (le nombre de décimales écrites fixe la tolérance)"""
    return abs(x - target) <= 0.5 * 10 ** (-decimals) + 1e-9


def _matches(mention: Mention, target: float) -> bool:
    if _close(mention.value, target, mention.decimals):
        return True
    return mention.percent and _close(mention.value / 100, target, mention.decimals + 2)


def _trivial(mention: Mention) -> bool:
    """Petits entiers (0 à 10) : numérotation, « 4 semaines »... on ne les contrôle pas."""
    return mention.decimals == 0 and not mention.percent and abs(mention.value) <= 10


def _source_numbers(question: Question, tool_calls: list) -> list[float]:
    """Tous les nombres que le LLM a le droit de citer : sorties d'outils, arguments, question."""
    text = json.dumps([[c.get("result"), c.get("error"), c["arguments"]] for c in tool_calls], ensure_ascii=False, default=str)
    numbers = [float(t) for t in _TOKEN.findall(text)]
    numbers += [m.value for m in extract_mentions(question.text)]
    return numbers


# --------------------------------------------------------------------------- notation d'une réponse
def score(question: Question, result, use_tools: bool = True) -> dict:
    answer = result.answer or ""
    mentions = extract_mentions(answer)
    sources = _source_numbers(question, result.tool_calls)
    untraceable = [
        m for m in mentions if not _trivial(m) and not any(_matches(m, v) for v in sources)
    ]
    is_trap = question.category == "trap"

    if is_trap:
        declined = any(marker in answer.lower() for marker in DECLINE_MARKERS)
        correct = declined and not untraceable
        tool_ok = float("nan")
    else:
        correct = all(any(_matches(m, e) for m in mentions) for e in question.expected)
        tool_ok = (
            float(any(c["ok"] and c["name"] in question.expected_tools for c in result.tool_calls))
            if use_tools else float("nan")
        )
    return {
        "correct": bool(correct),
        "tool_ok": tool_ok,
        "untraceable": len(untraceable),
        "has_untraceable": bool(untraceable),
        "n_tool_calls": len(result.tool_calls),
        "turns": result.turns,
        "stopped_reason": result.stopped_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": result.seconds,
        "error": "",
        "answer": answer,
    }


# --------------------------------------------------------------------------- exécution
def run_eval(conn, adapter_for, questions, use_tools=True, n_runs=3, log_path=None, max_turns=6, progress=None):
    """Pose chaque question `n_runs` fois. `adapter_for(question)` renvoie l'adaptateur LLM à utiliser.

    Une erreur d'API (réseau, quota...) est enregistrée dans la colonne `error` et n'arrête pas l'évaluation.
    """
    condition = "avec_outils" if use_tools else "sans_outils"
    rows, done = [], 0
    for run in range(1, n_runs + 1):
        for q in questions:
            try:
                agent = Agent(conn, adapter_for(q), use_tools=use_tools, max_turns=max_turns, log_path=log_path)
                row = score(q, agent.ask(q.text), use_tools)
            except Exception as error:  # noqa: BLE001 - on veut finir l'évaluation quoi qu'il arrive
                row = {
                    "correct": False, "tool_ok": float("nan"), "untraceable": 0, "has_untraceable": False,
                    "n_tool_calls": 0, "turns": 0, "stopped_reason": "error", "input_tokens": 0,
                    "output_tokens": 0, "seconds": 0.0, "answer": "",
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
            rows.append({"condition": condition, "run": run, "id": q.id, "category": q.category,
                         "hidden": q.hidden, **row})
            done += 1
            if progress:
                progress(done)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Un tableau par condition et par catégorie. Les lignes en erreur d'API sont comptées à part."""
    out = []
    frames = [df.assign(category="tout"), df]
    for frame in frames:
        for (condition, category), group in frame.groupby(["condition", "category"], sort=False):
            ok = group[group["error"] == ""]
            out.append({
                "condition": condition,
                "category": category,
                "n": len(ok),
                "erreurs_api": len(group) - len(ok),
                "exactitude": ok["correct"].mean() if len(ok) else float("nan"),
                "bon_outil": ok["tool_ok"].mean() if len(ok) else float("nan"),
                "avec_nombre_non_tracable": ok["has_untraceable"].mean() if len(ok) else float("nan"),
                "tokens_entree": ok["input_tokens"].mean() if len(ok) else float("nan"),
                "tokens_sortie": ok["output_tokens"].mean() if len(ok) else float("nan"),
                "secondes": ok["seconds"].mean() if len(ok) else float("nan"),
            })
    return pd.DataFrame(out).round(3)


# --------------------------------------------------------------------------- ligne de commande
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Évalue le copilote sur 30 questions à réponse connue.")
    parser.add_argument("--provider", default="demo", choices=["demo", "anthropic", "azure", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=3, help="Nombre de fois où chaque question est posée.")
    parser.add_argument("--split", default="dev", choices=["dev", "hidden", "all"])
    parser.add_argument("--conditions", default="both", choices=["both", "avec", "sans"])
    parser.add_argument("--limit", type=int, help="N'utiliser que les N premières questions (essai).")
    parser.add_argument("--db", default="data/sc.db")
    parser.add_argument("--out", default="eval/results/run.csv")
    parser.add_argument("--yes", action="store_true", help="Ne pas demander confirmation.")
    args = parser.parse_args(argv)

    if args.runs < 1:
        print("--runs doit être au moins 1.")
        return 1
    if not Path(args.db).exists():
        print(f"Base introuvable : {args.db}. Lance d'abord : python -m sc_copilot.data_gen")
        return 1
    try:
        shared = make_adapter(args.provider, args.model)
    except RuntimeError as error:
        print(error)
        return 1
    adapter_for = (lambda q: make_adapter("demo")) if args.provider == "demo" else (lambda q: shared)

    conn = sqlite3.connect(args.db)
    questions = build_questions(conn)
    if args.split != "all":
        questions = [q for q in questions if q.hidden == (args.split == "hidden")]
    if args.limit:
        questions = questions[: args.limit]
    conditions = ["avec", "sans"] if args.conditions == "both" else [args.conditions]

    conversations = len(questions) * args.runs * len(conditions)
    print(f"{len(questions)} questions ({args.split}) x {args.runs} passage(s) x {len(conditions)} condition(s) = {conversations} conversations.")
    if args.provider == "demo":
        print("MODE DÉMO : le faux LLM répond la même chose à tout. Les résultats ne veulent rien dire, c'est un test du circuit.")
    else:
        calls = len(questions) * args.runs * (3 if "avec" in conditions else 0) + (len(questions) * args.runs if "sans" in conditions else 0)
        print(f"Environ {calls} appels à l'API, facturés par ton fournisseur. Commence avec --limit 5 --runs 1.")
        if args.split != "dev":
            print("ATTENTION : tu lances les questions CACHÉES. À faire une seule fois, à la fin.")
        if not args.yes and input("Continuer ? [o/N] ").strip().lower() not in ("o", "oui", "y", "yes"):
            print("Annulé.")
            return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".jsonl")
    frames = []
    for name in conditions:
        frames.append(
            run_eval(conn, adapter_for, questions, use_tools=CONDITIONS[name], n_runs=args.runs, log_path=log_path,
                     progress=lambda n: print(f"  {n} conversations faites", end="\r"))
        )
    conn.close()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print("\n" + summarize(df).to_string(index=False))
    print(f"\nDétail : {out}   Journal complet : {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
