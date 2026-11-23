# sc-copilot

Prototype de **copilote IA pour la décision en supply chain** (réapprovisionnement, suivi de performance, planification). Un LLM comprend la question et appelle des outils testés. **Le LLM ne calcule jamais** : chaque chiffre vient d'un code vérifié, ce qui limite les chiffres inventés.

> **État : prototype en cours.** Données **fictives**. Les calculs sont testés (plus de 250 tests automatiques). L'évaluation avec un vrai LLM est préparée mais **pas encore exécutée** : ce dépôt ne contient donc aucun chiffre de précision du copilote.

## Pourquoi ce projet

Un LLM seul peut produire un chiffre plausible mais faux, sans le signaler. L'idée est de séparer les rôles :

```
Question en français
        |
   [ LLM ]  comprend la question, choisit l'outil, rédige la réponse
        |
   [ Outils ]  calculent (OTIF, prévision, stock de sécurité, planification...)
        |
   [ Base de données ]
```

Le périmètre est volontairement restreint pour pouvoir être testé sérieusement. Le copilote joue le rôle d'**analyste** (il prépare une décision), pas de décideur ni de consultant.

## Ce que contient le dépôt

| Module | Rôle |
|---|---|
| `data_gen.py` | Base fictive reproductible : 300 pièces, 20 fournisseurs, 3 ans de demande, 3 perturbations fournisseurs, commandes ouvertes en option. |
| `dataio.py` | Lecture et **vérification** de 5 fichiers CSV (format standard), puis création de la base. Permet de brancher d'autres données. |
| `kpi.py` | OTIF, couverture de stock, pièces à risque de rupture, stock qui expire. |
| `forecast.py` | Classification de la demande (Syntetos-Boylan), moyenne mobile, lissage exponentiel, Croston SBA, comparaison des méthodes. |
| `inventory.py` | Stock de sécurité, point de commande, quantité à commander (loi normale ou bootstrap), niveau de service atteint. |
| `analysis.py` | Principaux écarts de livraison, risque de retard des commandes ouvertes, résumé de performance. |
| `scheduling.py` | Ordonnancement avec le solveur CP-SAT (OR-Tools) : une machine, flow shop, job shop. |
| `tools.py` | Les 10 outils exposés au LLM, avec validation stricte des arguments et SQL en lecture seule. |
| `agent.py` | Boucle question, outil, résultat, réponse. Adaptateurs Claude, Azure OpenAI, OpenAI, et un faux LLM de démonstration. |
| `evaluation.py` | 30 questions à réponse vérifiée, mesure avec et sans outils. |
| `app/streamlit_app.py` | Interface web. |

### Les 10 outils

`get_otif`, `list_parts_at_risk`, `list_expiring_stock`, `forecast_demand`, `recommend_reorder`, `top_deviations`, `delay_risk`, `performance_summary`, `schedule_tasks`, `run_sql_readonly`.

Chaque outil renvoie un résultat structuré, ou une erreur explicite que le LLM peut lire. Le SQL est limité aux requêtes `SELECT` sur une liste de tables autorisées, avec plafond de lignes et délai maximal.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e .
pip install pytest streamlit ortools
```

Pour utiliser un vrai LLM, installer aussi `anthropic` ou `openai`, et fournir la clé **dans une variable d'environnement** (jamais dans un fichier du dépôt).

## Utilisation

```bash
python -m sc_copilot.data_gen --open-orders 60     # crée data/sc.db (données fictives)
pytest                                             # lance les tests
python -m sc_copilot.agent --demo                  # démonstration avec le faux LLM
streamlit run app/streamlit_app.py                 # interface web
python -m sc_copilot.analysis                      # écarts et résumé de performance
python -m sc_copilot.scheduling                    # exemple d'ordonnancement
```

### Utiliser d'autres données (CSV)

```bash
python -m sc_copilot.dataio template mes_csv       # écrit 5 fichiers modèles
python -m sc_copilot.dataio check mes_csv          # vérifie les fichiers, sans rien créer
python -m sc_copilot.dataio build mes_csv --db data/sc.db
```

Le vérificateur signale précisément les problèmes : colonne absente, date invalide, clé en double, pièce inconnue, réception avant la commande. Il accepte les exports Excel français (séparateur `;`, virgule décimale, dates `JJ/MM/AAAA`).

## Évaluation (préparée, non exécutée)

- **30 questions** : 8 KPI, 6 prévisions, 6 politiques de stock, 6 requêtes SQL, 4 pièges (hors périmètre, à refuser).
- **Réponses attendues calculées indépendamment** par SQL direct, puis comparées aux outils par un test.
- **20 questions de développement, 10 cachées** : les cachées ne servent qu'à une mesure finale.
- **Deux conditions** : avec outils et sans outils. Mesures : réponse juste, bon outil appelé, nombres non traçables (qui ne viennent d'aucun outil), jetons, secondes, erreurs d'API.

Limites à connaître :
- Sans outils, le LLM ne peut pas connaître les données : cette condition est un point de comparaison faible.
- L'évaluation mesure le **choix des outils et la fidélité de la recopie des chiffres**, pas la qualité des prévisions.
- La notation des pièges repose sur des mots-clés ; une relecture manuelle d'un échantillon est nécessaire.
- Une coïncidence numérique reste possible.

Les nouveaux outils (`top_deviations`, `delay_risk`, `performance_summary`, `schedule_tasks`) sont testés, mais ne sont pas encore couverts par les 30 questions.

## Limites connues

- Données **fictives** : les tests prouvent que le code est correct, pas que les prévisions sont bonnes sur des données réelles.
- `delay_risk` ne compte pas le stock déjà en commande : il peut surestimer le risque.
- `schedule_tasks` reçoit les données du problème de l'utilisateur ; il ne les lit pas dans la base.
- Aucun résultat avec un vrai LLM pour l'instant.

## Suite prévue

1. Étendre l'évaluation aux nouveaux outils et l'exécuter avec un vrai LLM (dev, puis une seule passe sur les questions cachées).
2. Publier les résultats ici, y compris s'ils sont décevants.
3. Adapter le copilote à des données industrielles réelles via l'import CSV, après validation des outils avec les utilisateurs métier.
