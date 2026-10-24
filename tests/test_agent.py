import json
import sqlite3
from types import SimpleNamespace as NS

import pytest

from sc_copilot import kpi
from sc_copilot.agent import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_TOOLS,
    Agent,
    AnthropicAdapter,
    DemoAdapter,
    OpenAIAdapter,
    ToolCall,
    Turn,
    make_adapter,
    main,
)
from sc_copilot.data_gen import generate
from sc_copilot.tools import DATASET_TODAY


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db = generate(tmp_path_factory.mktemp("db") / "sc.db")
    c = sqlite3.connect(db)
    yield c
    c.close()


class ScriptedAdapter:
    """Faux LLM de test : renvoie les tours prévus, dans l'ordre, et garde ce qu'il a reçu."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_tools = []
        self.seen_messages = []

    def initial_messages(self, system, question):
        return [{"role": "user", "content": question}]

    def complete(self, system, messages, tools):
        self.seen_tools.append(tools)
        self.seen_messages.append(json.dumps(messages, ensure_ascii=False, default=str))
        return self.turns.pop(0)

    def tool_result_messages(self, executed):
        return [{"role": "tool", "content": json.dumps(o, ensure_ascii=False)} for _, o in executed]


def tool_turn(*calls):
    return Turn("", [ToolCall(f"id{i}", n, a) for i, (n, a) in enumerate(calls)], {"role": "assistant", "content": "x"})


def final_turn(text="Voilà."):
    return Turn(text, [], {"role": "assistant", "content": text})


# ---- la boucle
def test_une_demande_d_outil_puis_la_reponse(conn):
    adapter = ScriptedAdapter([tool_turn(("get_otif", {})), final_turn("OTIF ok")])
    result = Agent(conn, adapter).ask("Quel est l'OTIF ?")
    assert result.answer == "OTIF ok" and result.stopped_reason == "final" and result.turns == 2
    assert len(result.tool_calls) == 1 and result.tool_calls[0]["ok"]
    assert result.tool_calls[0]["result"]["otif"] == pytest.approx(kpi.otif(conn), abs=1e-4)
    # le LLM a bien reçu le résultat de l'outil au 2e tour
    assert "otif" in adapter.seen_messages[1]


def test_reponse_directe_sans_outil(conn):
    result = Agent(conn, ScriptedAdapter([final_turn("Bonjour")])).ask("Salut")
    assert result.tool_calls == [] and result.turns == 1


def test_plusieurs_outils_dans_un_meme_tour(conn):
    adapter = ScriptedAdapter(
        [tool_turn(("get_otif", {}), ("recommend_reorder", {"part_id": 1})), final_turn()]
    )
    result = Agent(conn, adapter).ask("Les deux")
    assert [c["name"] for c in result.tool_calls] == ["get_otif", "recommend_reorder"]


def test_erreur_d_outil_renvoyee_au_llm_qui_se_corrige(conn):
    adapter = ScriptedAdapter(
        [
            tool_turn(("forecast_demand", {"part_id": 99999})),   # pièce inconnue
            tool_turn(("forecast_demand", {"part_id": 5})),       # le LLM se corrige
            final_turn("Corrigé"),
        ]
    )
    result = Agent(conn, adapter).ask("Prévision ?")
    assert [c["ok"] for c in result.tool_calls] == [False, True]
    assert "introuvable" in adapter.seen_messages[1].lower() or "aucune demande" in adapter.seen_messages[1].lower()
    assert result.answer == "Corrigé"


def test_outil_inconnu_ne_plante_pas(conn):
    adapter = ScriptedAdapter([tool_turn(("supprimer_tout", {})), final_turn("Désolé")])
    result = Agent(conn, adapter).ask("Fais n'importe quoi")
    assert result.tool_calls[0]["ok"] is False and "inconnu" in result.tool_calls[0]["error"]


def test_arguments_en_texte_json(conn):
    adapter = ScriptedAdapter([tool_turn(("get_otif", '{"supplier_id": 4}')), final_turn()])
    assert Agent(conn, adapter).ask("OTIF 4").tool_calls[0]["result"]["scope"] == "fournisseur 4"


def test_garde_fou_max_turns(conn):
    adapter = ScriptedAdapter([tool_turn(("get_otif", {})) for _ in range(10)])
    result = Agent(conn, adapter, max_turns=3).ask("Boucle infinie")
    assert result.stopped_reason == "max_turns" and result.turns == 3 and len(adapter.turns) == 7


def test_sans_outils_aucun_menu_envoye(conn):
    adapter = ScriptedAdapter([final_turn("Je ne sais pas")])
    agent = Agent(conn, adapter, use_tools=False)
    agent.ask("OTIF ?")
    assert adapter.seen_tools == [[]]
    assert agent.system_prompt == SYSTEM_PROMPT_NO_TOOLS


def test_avec_outils_le_menu_est_envoye(conn):
    adapter = ScriptedAdapter([final_turn()])
    agent = Agent(conn, adapter)
    agent.ask("Question")
    assert len(adapter.seen_tools[0]) == 6
    assert agent.system_prompt == SYSTEM_PROMPT and DATASET_TODAY in SYSTEM_PROMPT


def test_les_tokens_sont_additionnes(conn):
    t1 = tool_turn(("get_otif", {}))
    t1.input_tokens, t1.output_tokens = 100, 10
    t2 = final_turn()
    t2.input_tokens, t2.output_tokens = 150, 20
    result = Agent(conn, ScriptedAdapter([t1, t2])).ask("Q")
    assert (result.input_tokens, result.output_tokens) == (250, 30)


def test_journal_jsonl(conn, tmp_path):
    log = tmp_path / "logs" / "run.jsonl"
    agent = Agent(conn, ScriptedAdapter([final_turn("a"), final_turn("b")]), log_path=log)
    agent.ask("Q1")
    agent.ask("Q2")
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [l["question"] for l in lines] == ["Q1", "Q2"] and lines[1]["answer"] == "b"


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_question_invalide(conn, bad):
    with pytest.raises(ValueError):
        Agent(conn, ScriptedAdapter([])).ask(bad)


def test_max_turns_invalide(conn):
    with pytest.raises(ValueError):
        Agent(conn, ScriptedAdapter([]), max_turns=0)


def test_resultat_trop_long_est_coupe():
    from sc_copilot.agent import MAX_TOOL_CHARS, _tool_content

    long = _tool_content({"ok": True, "result": "x" * (MAX_TOOL_CHARS * 2)})
    assert len(long) <= MAX_TOOL_CHARS + 100 and "coupé" in long
    short = _tool_content({"ok": True, "result": "court"})
    assert "coupé" not in short and json.loads(short)["result"] == "court"


# ---- adaptateur Claude, avec un faux client (aucun package ni clé nécessaire)
class FakeAnthropic:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=str)))
        return self.responses.pop(0)


def test_adaptateur_claude_boucle_complete(conn):
    client = FakeAnthropic(
        [
            NS(
                content=[
                    NS(type="text", text="Je vérifie."),
                    NS(type="tool_use", id="toolu_1", name="get_otif", input={"supplier_id": 3}),
                ],
                usage=NS(input_tokens=50, output_tokens=8),
            ),
            NS(content=[NS(type="text", text="L'OTIF est de 82,7 %.")], usage=NS(input_tokens=90, output_tokens=12)),
        ]
    )
    result = Agent(conn, AnthropicAdapter(client, "claude-haiku-5-5")).ask("OTIF du fournisseur 3 ?")
    assert result.answer == "L'OTIF est de 82,7 %." and (result.input_tokens, result.output_tokens) == (140, 20)

    first, second = client.calls
    assert first["model"] == "claude-haiku-5-5" and first["system"] == SYSTEM_PROMPT
    assert first["messages"] == [{"role": "user", "content": "OTIF du fournisseur 3 ?"}]
    assert {t["name"] for t in first["tools"]} >= {"get_otif", "run_sql_readonly"}
    assert all("input_schema" in t for t in first["tools"])
    # 2e appel : message du LLM (bloc tool_use) puis résultat en tool_result lié au bon identifiant
    assistant, tool_result = second["messages"][1], second["messages"][2]
    assert assistant["content"][1]["type"] == "tool_use" and assistant["content"][1]["id"] == "toolu_1"
    block = tool_result["content"][0]
    assert tool_result["role"] == "user" and block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1" and block["is_error"] is False
    assert json.loads(block["content"])["result"]["scope"] == "fournisseur 3"


def test_adaptateur_claude_erreur_marquee(conn):
    client = FakeAnthropic(
        [
            NS(content=[NS(type="tool_use", id="t1", name="get_otif", input={"supplier_id": 9999})], usage=None),
            NS(content=[NS(type="text", text="Fournisseur inconnu.")], usage=None),
        ]
    )
    Agent(conn, AnthropicAdapter(client, "m")).ask("OTIF du 9999 ?")
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_adaptateur_claude_sans_outils_et_temperature(conn):
    client = FakeAnthropic([NS(content=[NS(type="text", text="?")], usage=None)])
    Agent(conn, AnthropicAdapter(client, "m", temperature=0.0), use_tools=False).ask("Q")
    assert "tools" not in client.calls[0] and client.calls[0]["temperature"] == 0.0


# ---- adaptateur OpenAI / Azure, avec un faux client
class FakeOpenAI:
    def __init__(self, messages):
        self.messages, self.calls = list(messages), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=str)))
        return NS(choices=[NS(message=self.messages.pop(0))], usage=NS(prompt_tokens=40, completion_tokens=6))


def test_adaptateur_openai_boucle_complete(conn):
    tool_call = NS(id="call_1", function=NS(name="recommend_reorder", arguments='{"part_id": 1}'))
    client = FakeOpenAI(
        [NS(content=None, tool_calls=[tool_call]), NS(content="Commander 32 unités.", tool_calls=None)]
    )
    result = Agent(conn, OpenAIAdapter(client, "mon-deploiement")).ask("Que faire pour la pièce 1 ?")
    assert result.answer == "Commander 32 unités." and result.tool_calls[0]["result"]["order_quantity"] == 32

    first, second = client.calls
    assert first["model"] == "mon-deploiement"
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert all(t["type"] == "function" for t in first["tools"])
    assistant, tool_msg = second["messages"][2], second["messages"][3]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"part_id": 1}'
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"])["ok"] is True


def test_adaptateur_openai_reponse_sans_outil_pas_de_cle_tool_calls(conn):
    client = FakeOpenAI([NS(content="Direct.", tool_calls=None)])
    agent = Agent(conn, OpenAIAdapter(client, "m"), use_tools=False)
    agent.ask("Q")
    assert "tools" not in client.calls[0]


# ---- démo, création des adaptateurs, ligne de commande
def test_demo_adapter_donne_les_memes_chiffres_que_les_outils(conn):
    result = Agent(conn, DemoAdapter()).ask("N'importe quoi")
    assert f"{kpi.otif(conn):.1%}" in result.answer and "point de commande" in result.answer
    assert len(result.tool_calls) == 2 and all(c["ok"] for c in result.tool_calls)


def test_make_adapter_cle_absente(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY", "SC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for provider in ("anthropic", "azure", "openai"):
        with pytest.raises(RuntimeError):
            make_adapter(provider, "m")


def test_make_adapter_azure_sans_modele(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://exemple.openai.azure.com")
    monkeypatch.delenv("SC_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="déploiement"):
        make_adapter("azure")


def test_make_adapter_fournisseur_inconnu():
    with pytest.raises(ValueError):
        make_adapter("magie")


def test_ligne_de_commande_demo(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data" / "sc.db")
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "outil get_otif" in out and "Réponse :" in out and "Mode démo" in out


def test_ligne_de_commande_base_absente(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--demo"]) == 1
    assert "data_gen" in capsys.readouterr().out
