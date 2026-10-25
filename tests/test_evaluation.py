import json
import sqlite3

import pandas as pd
import pytest

from sc_copilot import evaluation
from sc_copilot.agent import AgentResult, Turn, ToolCall
from sc_copilot.data_gen import generate
from sc_copilot.evaluation import (
    Mention,
    _matches,
    build_questions,
    extract_mentions,
    main,
    reference_value,
    run_eval,
    score,
    summarize,
)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = generate(tmp_path_factory.mktemp("db") / "sc.db")
    c = sqlite3.connect(db)
    yield c
    c.close()


@pytest.fixture(scope="module")
def questions(conn):
    return build_questions(conn)


class OracleAdapter:
    """Faux LLM parfait : appelle l'outil de référence et recopie le résultat. Sans outils, il devine."""

    def __init__(self, question):
        self.q, self.step, self.outcome = question, 0, None

    def initial_messages(self, system, question):
        return [{"role": "user", "content": question}]

    def complete(self, system, messages, tools):
        self.step += 1
        if self.q.category == "trap":
            text = "Je ne peux pas répondre : aucun outil ne le permet."
        elif not tools:
            text = "Environ 42,5 unités."
        elif self.step == 1:
            tool, args, _ = self.q.reference
            return Turn("", [ToolCall("c1", tool, args)], {"role": "assistant", "content": "x"})
        else:
            value = self.outcome["result"]
            for key in self.q.reference[2]:
                value = value[key]
            text = f"Résultat : {value}"
        return Turn(text, [], {"role": "assistant", "content": text})

    def tool_result_messages(self, executed):
        self.outcome = executed[0][1]
        return [{"role": "user", "content": "r"}]


class FixedAdapter:
    def __init__(self, text):
        self.text = text

    def initial_messages(self, system, question):
        return [{"role": "user", "content": question}]

    def complete(self, system, messages, tools):
        return Turn(self.text, [], {"role": "assistant", "content": self.text})

    def tool_result_messages(self, executed):
        return []


class BrokenAdapter(FixedAdapter):
    def complete(self, system, messages, tools):
        raise RuntimeError("quota dépassé")


def fake_result(answer, tool_calls=()):
    return AgentResult(question="q", answer=answer, tool_calls=list(tool_calls), turns=2)


OTIF_CALL = {"name": "get_otif", "arguments": {}, "ok": True, "result": {"scope": "global", "otif": 0.736}}


# ---- lecture des nombres
@pytest.mark.parametrize(
    "text, expected",
    [
        ("73,6 %", [(73.6, 1, True)]),
        ("73.6%", [(73.6, 1, True)]),
        ("Le stock est de 12 345 unités", [(12345.0, 0, False)]),
        ("1 234,5", [(1234.5, 1, False)]),
        ("point de commande 26.0, stock 2", [(26.0, 1, False), (2.0, 0, False)]),
        ("au 29 décembre 2025", [(29.0, 0, False), (2025.0, 0, False)]),
        ("la date 2025-12-29 est ignorée", []),
        ("niveau 0,95", [(0.95, 2, False)]),
    ],
)
def test_extraction(text, expected):
    got = [(m.value, m.decimals, m.percent) for m in extract_mentions(text)]
    assert got == expected


def test_comparaison_a_l_arrondi_pres():
    assert _matches(Mention(73.6, 1, True), 0.7361)       # 73,6 % pour 0,7361
    assert _matches(Mention(74, 0, True), 0.7361)         # 74 % : arrondi acceptable
    assert not _matches(Mention(88.2, 1, True), 0.7361)
    assert _matches(Mention(1.77, 2, False), 1.771728)
    assert not _matches(Mention(1.8, 1, False), 1.771728 + 0.1)
    assert not _matches(Mention(60, 0, False), 61)        # « environ 60 » n'est pas 61


# ---- les questions
def test_structure_des_questions(questions):
    assert len(questions) == 30
    counts = {c: sum(q.category == c for q in questions) for c in ("kpi", "forecast", "policy", "sql", "trap")}
    assert counts == {"kpi": 8, "forecast": 6, "policy": 6, "sql": 6, "trap": 4}
    assert len({q.id for q in questions}) == 30 and len({q.text for q in questions}) == 30
    assert sum(q.hidden for q in questions) == 10
    assert all(q.expected_tools and q.expected and q.reference for q in questions if q.category != "trap")
    assert all(not q.expected and not q.reference for q in questions if q.category == "trap")


def test_reponses_attendues_retrouvees_par_les_outils(conn, questions):
    """La réponse calculée par SQL direct doit être celle des outils : sinon la question est fausse."""
    for q in (q for q in questions if q.category != "trap"):
        assert reference_value(conn, q) == pytest.approx(q.expected[0], abs=1e-3), q.id  # les outils arrondissent à 3 décimales


def test_pas_de_reponse_attendue_trop_petite(questions):
    """Un petit entier attendu serait trouvé par hasard dans n'importe quelle réponse."""
    for q in questions:
        for e in q.expected:
            assert e >= 11 or not float(e).is_integer(), (q.id, e)


# ---- notation
def test_reponse_exacte_avec_nombre_tracable(questions):
    q = questions[0]  # OTIF global
    row = score(q, fake_result(f"L'OTIF global est de {q.expected[0]:.1%}.", [dict(OTIF_CALL, result={"otif": q.expected[0]})]))
    assert row["correct"] and row["tool_ok"] == 1.0 and row["untraceable"] == 0


def test_nombre_invente_est_detecte(questions):
    q = questions[0]
    row = score(q, fake_result("L'OTIF est de 73,6 % et le coût sera de 845 euros.", [OTIF_CALL]))
    assert row["untraceable"] == 1 and row["has_untraceable"]


def test_sans_outils_un_nombre_devine_est_non_tracable(questions):
    row = score(questions[0], fake_result("Le taux OTIF est de 88,2 %."), use_tools=False)
    assert not row["correct"] and row["untraceable"] == 1 and row["tool_ok"] != row["tool_ok"]  # NaN


def test_dates_et_petits_entiers_ne_comptent_pas(questions):
    call = {"name": "list_expiring_stock", "arguments": {}, "ok": True, "result": {"as_of": "2025-12-29", "total": 40}}
    row = score(questions[0], fake_result("Au 29 décembre 2025, il y a 40 pièces, soit 3 lots.", [call]))
    assert row["untraceable"] == 0


def test_piege_refus_propre(questions):
    trap = next(q for q in questions if q.category == "trap")
    assert score(trap, fake_result("Je ne peux pas prévoir le cours de l'acier : aucun outil ne le permet."))["correct"]


def test_piege_chiffre_invente(questions):
    trap = next(q for q in questions if q.category == "trap")
    assert not score(trap, fake_result("Le cours de l'acier sera de 845 euros la tonne."))["correct"]
    assert not score(trap, fake_result("Voici ma réponse."))["correct"]


def test_mauvais_outil(questions):
    q = next(q for q in questions if q.category == "forecast")
    row = score(q, fake_result("Résultat 12,5", [OTIF_CALL]))
    assert row["tool_ok"] == 0.0


# ---- exécution complète avec des faux LLM
def test_oracle_avec_outils_tout_juste(conn, questions):
    df = run_eval(conn, OracleAdapter, questions, use_tools=True, n_runs=1)
    assert len(df) == 30 and (df["error"] == "").all()
    assert df["correct"].all(), df[~df["correct"]][["id", "answer"]].to_string()
    assert not df["has_untraceable"].any()
    assert (df[df["category"] != "trap"]["tool_ok"] == 1.0).all()


def test_oracle_sans_outils_devine_faux(conn, questions):
    df = run_eval(conn, OracleAdapter, questions, use_tools=False, n_runs=1)
    non_trap = df[df["category"] != "trap"]
    assert not non_trap["correct"].any() and non_trap["has_untraceable"].all()


def test_plusieurs_passages_et_journal(conn, questions, tmp_path):
    log = tmp_path / "run.jsonl"
    df = run_eval(conn, OracleAdapter, questions[:3], use_tools=True, n_runs=2, log_path=log)
    assert len(df) == 6 and sorted(df["run"].unique()) == [1, 2]
    assert len(log.read_text(encoding="utf-8").splitlines()) == 6


def test_erreur_d_api_n_arrete_pas_l_evaluation(conn, questions):
    def adapter_for(q):
        return BrokenAdapter("x") if q.id == "kpi_1" else OracleAdapter(q)

    df = run_eval(conn, adapter_for, questions[:3], use_tools=True, n_runs=1)
    broken = df[df["id"] == "kpi_1"].iloc[0]
    assert "quota dépassé" in broken["error"] and not broken["correct"]
    assert (df[df["id"] != "kpi_1"]["error"] == "").all() and df[df["id"] != "kpi_1"]["correct"].all()


def test_progression(conn, questions):
    seen = []
    run_eval(conn, OracleAdapter, questions[:4], n_runs=1, progress=seen.append)
    assert seen == [1, 2, 3, 4]


def test_resume(conn, questions):
    good = run_eval(conn, OracleAdapter, questions[:6], use_tools=True, n_runs=1)
    bad = run_eval(conn, lambda q: BrokenAdapter("x") if q.id == "kpi_1" else OracleAdapter(q), questions[:2], use_tools=False, n_runs=1)
    table = summarize(pd.concat([good, bad], ignore_index=True))
    assert {"condition", "category", "n", "erreurs_api", "exactitude", "bon_outil", "avec_nombre_non_tracable"} <= set(table.columns)
    total = table[(table["category"] == "tout") & (table["condition"] == "avec_outils")].iloc[0]
    assert total["n"] == 6 and total["exactitude"] == 1.0 and total["bon_outil"] == 1.0
    sans = table[(table["category"] == "tout") & (table["condition"] == "sans_outils")].iloc[0]
    assert sans["erreurs_api"] == 1 and sans["n"] == 1


# ---- ligne de commande
def test_cli_demo(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    assert main(["--provider", "demo", "--runs", "1", "--limit", "3", "--out", "res/out.csv"]) == 0
    out = capsys.readouterr().out
    assert "MODE DÉMO" in out and "exactitude" in out
    assert (tmp_path / "res" / "out.csv").exists() and (tmp_path / "res" / "out.jsonl").exists()


def test_cli_separation_dev_et_cachees(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    main(["--runs", "1", "--conditions", "avec", "--out", "a.csv"])
    assert "20 questions (dev)" in capsys.readouterr().out
    main(["--runs", "1", "--conditions", "avec", "--split", "hidden", "--out", "b.csv"])
    assert "10 questions (hidden)" in capsys.readouterr().out


def test_cli_base_absente_et_parametres_invalides(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--db", "nulle_part.db"]) == 1 and "data_gen" in capsys.readouterr().out
    generate(tmp_path / "data" / "sc.db")
    assert main(["--runs", "0"]) == 1


def test_cli_cle_absente(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--provider", "anthropic"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_cli_confirmation_demandee_avant_les_appels_payants(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    monkeypatch.setattr(evaluation, "make_adapter", lambda provider, model=None: FixedAdapter("Je ne peux pas."))
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["--provider", "anthropic", "--runs", "1", "--limit", "2", "--out", "x.csv"]) == 0
    out = capsys.readouterr().out
    assert "facturés" in out and "Annulé" in out and not (tmp_path / "x.csv").exists()
    monkeypatch.setattr("builtins.input", lambda _: "o")
    assert main(["--provider", "anthropic", "--runs", "1", "--limit", "2", "--out", "x.csv"]) == 0
    assert (tmp_path / "x.csv").exists()
