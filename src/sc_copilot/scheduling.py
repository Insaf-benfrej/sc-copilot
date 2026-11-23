"""Ordonnancement avec le solveur CP-SAT (OR-Tools) : dans quel ordre exécuter des tâches sur des ressources.

Modèle général (job shop) : chaque « job » est une suite ordonnée d'opérations, chacune sur une machine
(ou un poste, un quai, une équipe...) pendant une durée entière. Une machine ne fait qu'une opération à la
fois. Ce modèle couvre une seule machine, un flow shop (tous les jobs passent par les mêmes machines dans le
même ordre) et un job shop (chaque job a son propre parcours).

Objectifs :
- "makespan"        : finir le plus tôt possible l'ensemble des jobs ;
- "total_tardiness" : minimiser la somme des retards par rapport aux dates dues (`due` obligatoire).

Un solveur renvoie parfois seulement une bonne solution, sans preuve d'optimalité : le champ `status` le dit
("optimal" ou "feasible") et `lower_bound` donne la meilleure borne connue.

Utilisation :
    python -m sc_copilot.scheduling
"""

from __future__ import annotations

MAX_JOBS = 30
MAX_OPERATIONS_PER_JOB = 10
MAX_MACHINES = 10
MAX_DURATION = 1000
MAX_TIME_LIMIT_S = 30.0


def _parse(jobs) -> list[dict]:
    """Vérifie et normalise les jobs. Lève ValueError avec un message précis."""
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs doit être une liste non vide.")
    if len(jobs) > MAX_JOBS:
        raise ValueError(f"Trop de jobs ({len(jobs)}) : maximum {MAX_JOBS}.")
    clean, seen, machines = [], set(), set()
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Le job n°{index} doit être un objet avec id et operations.")
        unknown = set(job) - {"id", "operations", "release", "due"}
        if unknown:
            raise ValueError(f"Job n°{index} : champ(s) inconnu(s) : {', '.join(sorted(unknown))}.")
        job_id = job.get("id")
        if not isinstance(job_id, (str, int)) or isinstance(job_id, bool) or not str(job_id).strip():
            raise ValueError(f"Le job n°{index} doit avoir un id (texte ou nombre).")
        job_id = str(job_id).strip()
        if len(job_id) > 30:
            raise ValueError(f"id trop long pour le job n°{index} (30 caractères maximum).")
        if job_id in seen:
            raise ValueError(f"id de job en double : {job_id}.")
        seen.add(job_id)

        operations = job.get("operations")
        if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS_PER_JOB:
            raise ValueError(f"Job {job_id} : operations doit être une liste de 1 à {MAX_OPERATIONS_PER_JOB} éléments.")
        ops = []
        for k, op in enumerate(operations, start=1):
            if not isinstance(op, dict) or set(op) - {"machine", "duration"} or "machine" not in op or "duration" not in op:
                raise ValueError(f"Job {job_id}, opération {k} : attendu {{machine, duration}}.")
            machine, duration = op["machine"], op["duration"]
            if not isinstance(machine, (str, int)) or isinstance(machine, bool) or not str(machine).strip():
                raise ValueError(f"Job {job_id}, opération {k} : machine invalide.")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration != int(duration):
                raise ValueError(f"Job {job_id}, opération {k} : duration doit être un nombre entier.")
            if not 1 <= duration <= MAX_DURATION:
                raise ValueError(f"Job {job_id}, opération {k} : duration doit être entre 1 et {MAX_DURATION}.")
            machines.add(str(machine).strip())
            ops.append({"machine": str(machine).strip(), "duration": int(duration)})

        parsed = {"id": job_id, "operations": ops, "release": 0, "due": None}
        for field in ("release", "due"):
            value = job.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 0:
                raise ValueError(f"Job {job_id} : {field} doit être un entier positif ou nul.")
            parsed[field] = int(value)
        clean.append(parsed)
    if len(machines) > MAX_MACHINES:
        raise ValueError(f"Trop de machines ({len(machines)}) : maximum {MAX_MACHINES}.")
    return clean


def solve_schedule(jobs, objective: str = "makespan", time_limit_s: float = 5.0) -> dict:
    """Calcule un ordonnancement. Voir la docstring du module. Lève ValueError si les données sont invalides."""
    if objective not in ("makespan", "total_tardiness"):
        raise ValueError("objective doit être 'makespan' ou 'total_tardiness'.")
    if isinstance(time_limit_s, bool) or not isinstance(time_limit_s, (int, float)) or not 0.1 <= time_limit_s <= MAX_TIME_LIMIT_S:
        raise ValueError(f"time_limit_s doit être entre 0,1 et {MAX_TIME_LIMIT_S:g} secondes.")
    parsed = _parse(jobs)
    if objective == "total_tardiness":
        missing = [j["id"] for j in parsed if j["due"] is None]
        if missing:
            raise ValueError(f"total_tardiness demande une date due (due) pour chaque job. Manquant : {', '.join(missing)}.")
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise ValueError("Le solveur n'est pas installé. Installe-le avec : pip install ortools") from error

    horizon = max(j["release"] for j in parsed) + sum(op["duration"] for j in parsed for op in j["operations"])
    model = cp_model.CpModel()
    starts, ends, by_machine = {}, {}, {}
    for j, job in enumerate(parsed):
        for k, op in enumerate(job["operations"]):
            start = model.new_int_var(0, horizon, f"s_{j}_{k}")
            end = model.new_int_var(0, horizon, f"e_{j}_{k}")
            interval = model.new_interval_var(start, op["duration"], end, f"i_{j}_{k}")
            starts[j, k], ends[j, k] = start, end
            by_machine.setdefault(op["machine"], []).append(interval)
            if k == 0:
                model.add(start >= job["release"])
            else:
                model.add(start >= ends[j, k - 1])
    for intervals in by_machine.values():
        model.add_no_overlap(intervals)

    job_end = [ends[j, len(job["operations"]) - 1] for j, job in enumerate(parsed)]
    if objective == "makespan":
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, job_end)
        model.minimize(makespan)
    else:
        delays = []
        for j, job in enumerate(parsed):
            delay = model.new_int_var(0, horizon, f"d_{j}")
            model.add(delay >= job_end[j] - job["due"])
            delays.append(delay)
        model.minimize(sum(delays))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = 1  # un seul fil : même données, même réponse
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    names = {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "feasible",
        cp_model.INFEASIBLE: "infeasible",
    }
    status_name = names.get(status, "unknown")
    result = {
        "status": status_name,
        "objective": objective,
        "solve_seconds": round(solver.wall_time, 3),
        "jobs": [],
        "machines": {},
    }
    if status_name not in ("optimal", "feasible"):
        return result

    machine_ops: dict[str, list] = {}
    total_tardiness = late_jobs = makespan_value = 0
    for j, job in enumerate(parsed):
        ops = []
        for k, op in enumerate(job["operations"]):
            s = solver.value(starts[j, k])
            ops.append({"machine": op["machine"], "start": s, "end": s + op["duration"]})
            machine_ops.setdefault(op["machine"], []).append({"job": job["id"], "start": s, "end": s + op["duration"]})
        end = ops[-1]["end"]
        tardiness = max(0, end - job["due"]) if job["due"] is not None else None
        if tardiness:
            total_tardiness += tardiness
            late_jobs += 1
        makespan_value = max(makespan_value, end)
        result["jobs"].append({"id": job["id"], "start": ops[0]["start"], "end": end, "due": job["due"],
                               "tardiness": tardiness, "operations": ops})
    result["machines"] = {m: sorted(items, key=lambda x: x["start"]) for m, items in sorted(machine_ops.items())}
    result["makespan"] = makespan_value
    result["total_tardiness"] = total_tardiness if objective == "total_tardiness" else None
    result["late_jobs"] = late_jobs if objective == "total_tardiness" else None
    result["objective_value"] = makespan_value if objective == "makespan" else total_tardiness
    result["lower_bound"] = round(solver.best_objective_bound, 3)
    return result


def check_schedule(jobs, result: dict) -> list[str]:
    """Vérificateur indépendant du solveur : renvoie la liste des règles violées (vide = ordonnancement valide)."""
    problems: list[str] = []
    parsed = _parse(jobs)
    plan = {item["id"]: item for item in result.get("jobs", [])}
    machine_slots: dict[str, list] = {}
    for job in parsed:
        item = plan.get(job["id"])
        if item is None or len(item["operations"]) != len(job["operations"]):
            problems.append(f"job {job['id']} : planification absente ou incomplète")
            continue
        previous_end = job["release"]
        for k, (op, placed) in enumerate(zip(job["operations"], item["operations"])):
            if placed["machine"] != op["machine"]:
                problems.append(f"job {job['id']} op {k + 1} : mauvaise machine")
            if placed["end"] - placed["start"] != op["duration"]:
                problems.append(f"job {job['id']} op {k + 1} : durée différente de celle demandée")
            if placed["start"] < previous_end:
                problems.append(f"job {job['id']} op {k + 1} : commence avant la fin de l'étape précédente ou avant la date de début")
            previous_end = placed["end"]
            machine_slots.setdefault(op["machine"], []).append((placed["start"], placed["end"], job["id"]))
    for machine, slots in machine_slots.items():
        slots.sort()
        for (s1, e1, j1), (s2, e2, j2) in zip(slots, slots[1:]):
            if s2 < e1:
                problems.append(f"machine {machine} : {j1} et {j2} se chevauchent")
    if result.get("jobs"):
        real = max(item["end"] for item in result["jobs"])
        if result.get("makespan") != real:
            problems.append("makespan annoncé différent de la fin réelle")
    return problems


if __name__ == "__main__":
    demo = [
        {"id": "A", "due": 6, "operations": [{"machine": "M1", "duration": 3}, {"machine": "M2", "duration": 2}]},
        {"id": "B", "due": 7, "operations": [{"machine": "M1", "duration": 2}, {"machine": "M2", "duration": 4}]},
    ]
    outcome = solve_schedule(demo)
    print(f"Statut : {outcome['status']}, makespan : {outcome['makespan']}")
    for machine, slots in outcome["machines"].items():
        print(machine, [(s["job"], s["start"], s["end"]) for s in slots])
    print("Violations :", check_schedule(demo, outcome) or "aucune")
