import itertools
import json
import random
import sqlite3
import sys

import pytest

pytest.importorskip("ortools")

from sc_copilot import scheduling
from sc_copilot.scheduling import check_schedule, solve_schedule
from sc_copilot.tools import TOOLS, call_tool


def job(job_id, *ops, due=None, release=None):
    data = {"id": job_id, "operations": [{"machine": m, "duration": d} for m, d in ops]}
    if due is not None:
        data["due"] = due
    if release is not None:
        data["release"] = release
    return data


# ---- cas calculables à la main
def test_flow_shop_deux_machines_regle_de_johnson():
    jobs = [job("A", ("M1", 3), ("M2", 2)), job("B", ("M1", 2), ("M2", 4))]
    result = solve_schedule(jobs)
    assert result["status"] == "optimal" and result["makespan"] == 8  # ordre B puis A ; A puis B donnerait 9
    assert [s["job"] for s in result["machines"]["M1"]] == ["B", "A"]
    assert check_schedule(jobs, result) == []


def test_une_machine_retards_calcules_a_la_main():
    # SPT : B(2) 0-2 (due 3), A(4) 2-6 (due 4, retard 2), C(3) 6-9 (due 9) -> retard total 2 ; aucun ordre ne fait mieux
    jobs = [job("A", ("M", 4), due=4), job("B", ("M", 2), due=3), job("C", ("M", 3), due=9)]
    result = solve_schedule(jobs, "total_tardiness")
    assert result["status"] == "optimal" and result["total_tardiness"] == 2 and result["late_jobs"] == 1
    assert result["objective_value"] == 2 and result["lower_bound"] == pytest.approx(2)


def test_date_de_debut_respectee():
    jobs = [job("A", ("M", 2), release=10)]
    result = solve_schedule(jobs)
    assert result["jobs"][0]["start"] == 10 and result["makespan"] == 12


def test_machine_utilisee_par_un_seul_job_a_la_fois():
    jobs = [job("A", ("M", 5)), job("B", ("M", 5)), job("C", ("M", 5))]
    assert solve_schedule(jobs)["makespan"] == 15


def test_jobs_independants_sur_machines_differentes():
    jobs = [job("A", ("M1", 5)), job("B", ("M2", 7))]
    assert solve_schedule(jobs)["makespan"] == 7


# ---- comparaison avec la force brute sur des petits cas aléatoires
def brute_single_machine_tardiness(durations, dues, releases):
    best = None
    for order in itertools.permutations(range(len(durations))):
        time, total = 0, 0
        for i in order:
            time = max(time, releases[i]) + durations[i]
            total += max(0, time - dues[i])
        best = total if best is None else min(best, total)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_retard_total_egal_a_la_force_brute(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 6)
    durations = [rng.randint(1, 9) for _ in range(n)]
    releases = [rng.choice([0, 0, rng.randint(0, 8)]) for _ in range(n)]
    dues = [releases[i] + rng.randint(1, 15) for i in range(n)]
    jobs = [job(f"J{i}", ("M", durations[i]), due=dues[i], release=releases[i]) for i in range(n)]
    result = solve_schedule(jobs, "total_tardiness")
    assert result["status"] == "optimal"
    assert result["total_tardiness"] == brute_single_machine_tardiness(durations, dues, releases)
    assert check_schedule(jobs, result) == []


def brute_flow_shop(times):
    best = None
    for order in itertools.permutations(range(len(times))):
        free = [0] * len(times[0])
        for i in order:
            t = 0
            for m, d in enumerate(times[i]):
                t = max(t, free[m]) + d
                free[m] = t
        best = free[-1] if best is None else min(best, free[-1])
    return best


@pytest.mark.parametrize("seed", range(10))
def test_flow_shop_3_machines_egal_a_la_force_brute(seed):
    # pour le makespan et au plus 3 machines, un ordre unique sur toutes les machines est optimal
    rng = random.Random(100 + seed)
    n = rng.randint(3, 5)
    times = [[rng.randint(1, 9) for _ in range(3)] for _ in range(n)]
    jobs = [job(f"J{i}", *[(f"M{m}", times[i][m]) for m in range(3)]) for i in range(n)]
    result = solve_schedule(jobs)
    assert result["status"] == "optimal" and result["makespan"] == brute_flow_shop(times)
    assert check_schedule(jobs, result) == []


@pytest.mark.parametrize("seed", range(8))
def test_job_shop_valide_et_au_dessus_des_bornes_simples(seed):
    rng = random.Random(500 + seed)
    machines = ["A", "B", "C"]
    jobs = []
    for i in range(4):
        route = rng.sample(machines, rng.randint(2, 3))
        jobs.append(job(f"J{i}", *[(m, rng.randint(1, 8)) for m in route]))
    result = solve_schedule(jobs)
    assert result["status"] == "optimal" and check_schedule(jobs, result) == []
    longest_job = max(sum(op["duration"] for op in j["operations"]) for j in jobs)
    load = {m: sum(op["duration"] for j in jobs for op in j["operations"] if op["machine"] == m) for m in machines}
    assert result["makespan"] >= max(longest_job, *load.values())


# ---- le vérificateur détecte bien les erreurs
@pytest.fixture()
def valid():
    jobs = [job("A", ("M1", 3), ("M2", 2)), job("B", ("M1", 2), ("M2", 4))]
    return jobs, solve_schedule(jobs)


def test_verificateur_detecte_chevauchement(valid):
    jobs, result = valid
    result["jobs"][0]["operations"][0].update(start=0, end=3)
    result["jobs"][1]["operations"][0].update(start=1, end=3)
    assert any("chevauchent" in p for p in check_schedule(jobs, result))


def test_verificateur_detecte_mauvaise_duree_et_ordre(valid):
    jobs, result = valid
    result["jobs"][0]["operations"][1].update(start=0, end=2)  # commence avant la fin de l'étape 1
    assert any("avant" in p for p in check_schedule(jobs, result))
    jobs, result = valid
    result["jobs"][0]["operations"][0]["end"] += 1
    assert any("durée" in p for p in check_schedule(jobs, result))


def test_verificateur_detecte_makespan_faux_et_job_absent(valid):
    jobs, result = valid
    result["makespan"] = 3
    assert any("makespan" in p for p in check_schedule(jobs, result))
    jobs, result = valid
    result["jobs"].pop()
    assert any("absente" in p for p in check_schedule(jobs, result))


# ---- données invalides
@pytest.mark.parametrize(
    "jobs, objective, message",
    [
        ([], "makespan", "liste non vide"),
        ("abc", "makespan", "liste non vide"),
        ([job("A", ("M", 1))] * 2, "makespan", "double"),
        ([{"id": "A"}], "makespan", "operations"),
        ([{"id": "A", "operations": []}], "makespan", "operations"),
        ([job("A", ("M", 0))], "makespan", "duration"),
        ([job("A", ("M", 2000))], "makespan", "duration"),
        ([job("A", ("M", 1.5))], "makespan", "entier"),
        ([job("A", ("M", True))], "makespan", "entier"),
        ([{"id": "A", "operations": [{"machine": "M"}]}], "makespan", "machine, duration"),
        ([job("A", ("M", 1), release=-1)], "makespan", "release"),
        ([{**job("A", ("M", 1)), "couleur": "rouge"}], "makespan", "inconnu"),
        ([job("A", ("M", 1))], "total_tardiness", "date due"),
        ([job("A", ("M", 1))], "vitesse", "objective"),
        ([job(f"J{i}", ("M", 1)) for i in range(31)], "makespan", "Trop de jobs"),
        ([job("A", *[(f"M{i}", 1) for i in range(10)]), job("B", ("M10", 1))], "makespan", "Trop de machines"),
    ],
)
def test_donnees_invalides(jobs, objective, message):
    with pytest.raises(ValueError, match=message):
        solve_schedule(jobs, objective)


@pytest.mark.parametrize("limit", [0, 0.01, 100, True, "5"])
def test_limite_de_temps_invalide(limit):
    with pytest.raises(ValueError, match="time_limit_s"):
        solve_schedule([job("A", ("M", 1))], time_limit_s=limit)


def test_resultat_reproductible():
    jobs = [job(f"J{i}", ("M1", i + 2), ("M2", 7 - i)) for i in range(5)]
    a, b = solve_schedule(jobs), solve_schedule(jobs)
    a.pop("solve_seconds"), b.pop("solve_seconds")
    assert a == b


def test_solveur_absent_message_clair(monkeypatch):
    monkeypatch.setitem(sys.modules, "ortools.sat.python", None)
    with pytest.raises(ValueError, match="pip install ortools"):
        solve_schedule([job("A", ("M", 1))])


# ---- l'outil
def test_outil_dans_le_menu():
    names = [t["name"] for t in TOOLS]
    assert "schedule_tasks" in names
    tool = next(t for t in TOOLS if t["name"] == "schedule_tasks")
    assert tool["parameters"]["required"] == ["jobs"]


def test_outil_resultat_json():
    conn = sqlite3.connect(":memory:")
    jobs = [job("A", ("M1", 3), ("M2", 2)), job("B", ("M1", 2), ("M2", 4))]
    res = call_tool(conn, "schedule_tasks", {"jobs": jobs, "objective": "makespan", "time_limit_s": 2})
    assert res["ok"] and res["result"]["makespan"] == 8
    json.dumps(res)
    res = call_tool(conn, "schedule_tasks", json.dumps({"jobs": jobs}))  # arguments en texte JSON
    assert res["ok"]


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"jobs": "pas une liste"},
        {"jobs": [{"id": "A", "operations": [{"machine": "M", "duration": -1}]}]},
        {"jobs": [job("A", ("M", 1))], "objective": "total_tardiness"},
        {"jobs": [job("A", ("M", 1))], "time_limit_s": 1000},
        {"jobs": [job("A", ("M", 1))], "couleur": "rouge"},
    ],
)
def test_outil_refuse_les_mauvais_arguments(args):
    res = call_tool(sqlite3.connect(":memory:"), "schedule_tasks", args)
    assert res["ok"] is False and res["error"]
