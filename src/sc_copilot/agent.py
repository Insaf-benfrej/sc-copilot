"""La boucle de l'agent : relie un LLM (modèle d'IA) à tes outils de calcul.

Le déroulé, pour une question :

    1. On envoie au LLM : la consigne (system prompt), la question et le menu des outils.
    2. Le LLM répond soit avec du texte (réponse finale), soit avec une demande d'outil.
    3. Si c'est une demande d'outil : TON programme exécute l'outil (call_tool),
       puis renvoie le résultat au LLM. On retourne à l'étape 2.
    4. On s'arrête à la réponse finale, ou après `max_turns` tours (garde-fou).

Le LLM ne calcule jamais : tous les chiffres viennent des outils.

Ce fichier ne dépend d'aucun fournisseur. Chaque fournisseur a un "adaptateur" qui
traduit les messages dans son format :
- AnthropicAdapter : API Claude              (pip install anthropic)
- OpenAIAdapter    : OpenAI ou Azure OpenAI  (pip install openai)
- DemoAdapter      : faux LLM qui suit un script, pour voir la boucle sans aucune clé

Utilisation :
    python -m sc_copilot.agent --demo
    python -m sc_copilot.agent --provider anthropic "Quel est le taux OTIF global ?"
    python -m sc_copilot.agent --provider azure "Que faire pour la pièce 1 ?"
"""

import argparse
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sc_copilot.tools import (
    DATASET_TODAY,
    SCHEMA,
    TOOLS,
    call_tool,
    to_anthropic_tools,
    to_openai_tools,
)

MAX_TOOL_CHARS = 20_000  # un résultat d'outil plus long est coupé avant d'être renvoyé au LLM

SYSTEM_PROMPT = f"""Tu es un assistant d'aide au réapprovisionnement de pièces, sur une base de données fictive de supply chain.

Règles :
1. Tu ne calcules jamais toi-même. Chaque nombre de ta réponse doit venir du résultat d'un outil.
2. Si aucun outil ne permet de répondre, ou si un outil renvoie une erreur que tu ne peux pas corriger, dis-le clairement au lieu de deviner.
3. Date de référence de la base : {DATASET_TODAY}. Les pièces sont numérotées de 1 à 300 et les fournisseurs de 1 à 20.
4. Réponds en français, en quelques phrases. Cite les outils utilisés et les hypothèses retenues (niveau de service, méthode).
5. Les données sont fictives : ne présente jamais un résultat comme une donnée réelle."""

SYSTEM_PROMPT_NO_TOOLS = f"""Tu es un assistant d'aide au réapprovisionnement de pièces, sur une base de données fictive de supply chain.
Tu n'as aucun outil et aucun accès aux données. Schéma de la base : {SCHEMA}.
Date de référence de la base : {DATASET_TODAY}. Réponds en français, en quelques phrases, du mieux que tu peux."""


# --------------------------------------------------------------------------- données échangées
@dataclass
class ToolCall:
    """Une demande d'outil faite par le LLM."""

    id: str
    name: str
    arguments: object  # dictionnaire, ou texte JSON (OpenAI)


@dataclass
class Turn:
    """Une réponse du LLM : du texte, des demandes d'outils, ou les deux."""

    text: str
    tool_calls: list
    assistant_message: dict  # le message à ajouter à la conversation, au format du fournisseur
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list = field(default_factory=list)  # [{"name", "arguments", "ok", "result"|"error"}]
    turns: int = 0
    stopped_reason: str = "final"  # "final" ou "max_turns"
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- l'agent
class Agent:
    def __init__(
        self,
        conn: sqlite3.Connection,
        adapter,
        use_tools: bool = True,
        max_turns: int = 6,
        system_prompt: str | None = None,
        log_path: str | Path | None = None,
    ):
        if max_turns < 1:
            raise ValueError("max_turns doit être au moins 1.")
        self.conn = conn
        self.adapter = adapter
        self.use_tools = use_tools
        self.max_turns = max_turns
        self.system_prompt = system_prompt or (SYSTEM_PROMPT if use_tools else SYSTEM_PROMPT_NO_TOOLS)
        self.log_path = Path(log_path) if log_path else None

    def ask(self, question: str) -> AgentResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question est vide.")
        start = time.monotonic()
        messages = self.adapter.initial_messages(self.system_prompt, question)
        result = AgentResult(question=question, answer="")
        tools = TOOLS if self.use_tools else []

        for _ in range(self.max_turns):
            turn = self.adapter.complete(self.system_prompt, messages, tools)
            result.turns += 1
            result.input_tokens += turn.input_tokens
            result.output_tokens += turn.output_tokens
            result.answer = turn.text
            messages.append(turn.assistant_message)

            if not turn.tool_calls:
                result.stopped_reason = "final"
                break

            executed = []
            for call in turn.tool_calls:
                outcome = call_tool(self.conn, call.name, call.arguments)
                executed.append((call, outcome))
                entry = {"name": call.name, "arguments": call.arguments, "ok": outcome["ok"]}
                entry["result" if outcome["ok"] else "error"] = outcome.get("result", outcome.get("error"))
                result.tool_calls.append(entry)
            messages.extend(self.adapter.tool_result_messages(executed))
        else:
            result.stopped_reason = "max_turns"

        result.seconds = round(time.monotonic() - start, 3)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        return result


def _tool_content(outcome: dict) -> str:
    """Résultat d'outil -> texte JSON, coupé s'il est trop long."""
    text = json.dumps(outcome, ensure_ascii=False)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + '... [résultat coupé : trop long]'
    return text


# --------------------------------------------------------------------------- adaptateur Claude
class AnthropicAdapter:
    """API Claude (Messages API). `client` = anthropic.Anthropic()."""

    def __init__(self, client, model: str, max_tokens: int = 1024, temperature: float | None = None):
        self.client, self.model = client, model
        self.max_tokens, self.temperature = max_tokens, temperature

    def initial_messages(self, system, question):
        return [{"role": "user", "content": question}]

    def complete(self, system, messages, tools) -> Turn:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if tools:
            kwargs["tools"] = to_anthropic_tools()
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self.client.messages.create(**kwargs)

        text, calls, content = "", [], []
        for block in response.content:
            if block.type == "text":
                text += block.text
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, block.input))
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        usage = getattr(response, "usage", None)
        return Turn(
            text=text,
            tool_calls=calls,
            assistant_message={"role": "assistant", "content": content},
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def tool_result_messages(self, executed) -> list:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": _tool_content(outcome),
                "is_error": not outcome["ok"],
            }
            for call, outcome in executed
        ]
        return [{"role": "user", "content": blocks}]


# --------------------------------------------------------------------------- adaptateur OpenAI / Azure
class OpenAIAdapter:
    """OpenAI ou Azure OpenAI (Chat Completions). `client` = OpenAI() ou AzureOpenAI(...).

    Avec Azure, `model` est le NOM DU DÉPLOIEMENT choisi dans le portail Azure.
    """

    def __init__(self, client, model: str, temperature: float | None = None):
        self.client, self.model, self.temperature = client, model, temperature

    def initial_messages(self, system, question):
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]

    def complete(self, system, messages, tools) -> Turn:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = to_openai_tools()
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments)
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        assistant = {"role": "assistant", "content": message.content or ""}
        if calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls
            ]
        usage = getattr(response, "usage", None)
        return Turn(
            text=message.content or "",
            tool_calls=calls,
            assistant_message=assistant,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def tool_result_messages(self, executed) -> list:
        return [
            {"role": "tool", "tool_call_id": call.id, "content": _tool_content(outcome)}
            for call, outcome in executed
        ]


# --------------------------------------------------------------------------- faux LLM de démonstration
class DemoAdapter:
    """Faux LLM : il suit un script fixe et NE COMPREND PAS la question.

    Il sert uniquement à voir la boucle fonctionner sans clé : il appelle deux outils,
    puis recopie leurs résultats dans une phrase.
    """

    def __init__(self):
        self._step = 0
        self._results = []

    def initial_messages(self, system, question):
        return [{"role": "user", "content": question}]

    def complete(self, system, messages, tools) -> Turn:
        self._step += 1
        if self._step == 1:
            calls = [ToolCall("demo-1", "get_otif", {}), ToolCall("demo-2", "recommend_reorder", {"part_id": 1})]
            return Turn("", calls, {"role": "assistant", "content": "[demande d'outils]"})
        otif = next(r["result"]["otif"] for c, r in self._results if c.name == "get_otif")
        rec = next(r["result"] for c, r in self._results if c.name == "recommend_reorder")
        text = (
            f"Taux OTIF global : {otif:.1%}. "
            f"Pièce 1 : point de commande {rec['reorder_point']}, stock actuel {rec['current_stock']}, "
            f"décision « {rec['decision']} », quantité à commander {rec['order_quantity']}."
        )
        return Turn(text, [], {"role": "assistant", "content": text})

    def tool_result_messages(self, executed) -> list:
        self._results = list(executed)
        return [{"role": "user", "content": "[résultats d'outils]"}]


# --------------------------------------------------------------------------- création de l'adaptateur
def make_adapter(provider: str, model: str | None = None):
    """Crée l'adaptateur voulu. Les clés se lisent dans des variables d'environnement, jamais dans un fichier."""
    model = model or os.environ.get("SC_MODEL")
    if provider == "demo":
        return DemoAdapter()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError('Variable ANTHROPIC_API_KEY absente. Dans PowerShell : $env:ANTHROPIC_API_KEY="ta-clé"')
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError("Installe le package avec : pip install anthropic") from error
        return AnthropicAdapter(anthropic.Anthropic(), model or "claude-haiku-5-5")
    if provider in ("azure", "openai"):
        needed = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"] if provider == "azure" else ["OPENAI_API_KEY"]
        missing = [name for name in needed if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Variable(s) d'environnement absente(s) : {', '.join(missing)}")
        if not model:
            raise RuntimeError("Indique le modèle (ou le nom du déploiement Azure) avec --model ou la variable SC_MODEL.")
        try:
            import openai
        except ImportError as error:
            raise RuntimeError("Installe le package avec : pip install openai") from error
        if provider == "azure":
            client = openai.AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            client = openai.OpenAI()
        return OpenAIAdapter(client, model)
    raise ValueError(f"Fournisseur inconnu : {provider}. Choix : demo, anthropic, azure, openai.")


# --------------------------------------------------------------------------- ligne de commande
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pose une question au copilote supply chain.")
    parser.add_argument("question", nargs="?", help="La question, entre guillemets.")
    parser.add_argument("--provider", default="demo", choices=["demo", "anthropic", "azure", "openai"])
    parser.add_argument("--demo", action="store_true", help="Raccourci pour --provider demo (aucune clé nécessaire).")
    parser.add_argument("--model", help="Modèle (ou nom du déploiement Azure).")
    parser.add_argument("--no-tools", action="store_true", help="LLM seul, sans outils (pour comparer).")
    parser.add_argument("--db", default="data/sc.db")
    parser.add_argument("--log", help="Fichier .jsonl où enregistrer chaque échange.")
    args = parser.parse_args(argv)

    provider = "demo" if args.demo else args.provider
    question = args.question or "Donne-moi le taux OTIF global et une recommandation pour la pièce 1."
    if not Path(args.db).exists():
        print(f"Base introuvable : {args.db}. Lance d'abord : python -m sc_copilot.data_gen")
        return 1
    try:
        adapter = make_adapter(provider, args.model)
    except RuntimeError as error:
        print(error)
        return 1

    with sqlite3.connect(args.db) as conn:
        result = Agent(conn, adapter, use_tools=not args.no_tools, log_path=args.log).ask(question)

    print(f"\nQuestion : {result.question}\n")
    for call in result.tool_calls:
        status = "ok" if call["ok"] else "ERREUR"
        print(f"  outil {call['name']} {call['arguments']} -> {status}")
    print(f"\nRéponse : {result.answer}\n")
    print(
        f"[{result.turns} tour(s), {result.input_tokens} tokens entrée, "
        f"{result.output_tokens} tokens sortie, {result.seconds} s, fin : {result.stopped_reason}]"
    )
    if provider == "demo":
        print("(Mode démo : ce faux LLM suit un script et ne lit pas ta question.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
