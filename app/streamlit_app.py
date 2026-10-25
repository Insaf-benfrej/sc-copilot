"""Interface web du copilote. Lancer avec : streamlit run app/streamlit_app.py"""
import json
import sqlite3
from pathlib import Path

import streamlit as st

from sc_copilot.agent import Agent, make_adapter

DB_PATH = Path("data/sc.db")
EXEMPLES = [
    "Quel est le taux OTIF global ?",
    "Quel est le point de commande de la pièce 1 pour un niveau de service de 95 % ?",
    "Combien de pièces sont à risque de rupture ?",
    "Quel sera le cours de l'acier le mois prochain ?",
]

st.set_page_config(page_title="sc-copilot", page_icon="📦")
st.title("sc-copilot")
st.caption("Copilote de décisions de réapprovisionnement. Le LLM ne calcule rien : il appelle des outils testés. Données fictives.")

if not DB_PATH.exists():
    st.error("Base introuvable. Lance d'abord : python -m sc_copilot.data_gen")
    st.stop()

with st.sidebar:
    provider = st.selectbox("Fournisseur LLM", ["demo", "anthropic", "azure", "openai"])
    model = st.text_input("Modèle (facultatif)") or None
    use_tools = st.checkbox("Utiliser les outils", value=True)
    if provider == "demo":
        st.warning("Mode démo : le faux LLM répond toujours la même chose (OTIF global, pièce 1). Il teste le circuit, rien d'autre.")

question = st.text_input("Ta question")
st.write("Exemples :")
for ex in EXEMPLES:
    if st.button(ex):
        question = ex

if question:
    try:
        adapter = make_adapter(provider, model)
        conn = sqlite3.connect(DB_PATH)
        try:
            result = Agent(conn, adapter, use_tools=use_tools).ask(question)
        finally:
            conn.close()
    except Exception as error:  # clé absente, quota, réseau...
        st.error(f"Erreur : {error}")
    else:
        st.subheader("Réponse")
        st.write(result.answer)
        st.caption(f"{result.turns} tour(s) · {result.seconds} s · arrêt : {result.stopped_reason}")
        st.subheader("Outils appelés")
        if not result.tool_calls:
            st.write("Aucun.")
        for call in result.tool_calls:
            with st.expander(f"{'✅' if call['ok'] else '❌'} {call['name']}"):
                st.code(json.dumps(call, ensure_ascii=False, indent=2, default=str), language="json")
