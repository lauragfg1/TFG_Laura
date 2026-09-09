"""Fuente única de las preguntas del dataset, compartida por los tres frameworks.

launch_debate.py, launch_single.py, autogen_debate.py y crewai_debate.py importan
estas funciones en vez de leer index.txt o reimplementar la iteración por su cuenta,
para garantizar que los tres procesan exactamente las mismas preguntas en el mismo
orden (antes dependía de que index.txt y los --limit/--start manuales coincidieran
por convención, sin garantía a nivel de código).

Se excluye explícitamente index.txt del glob: al ser la concatenación de todos los
demás archivos, incluirlo como "archivo más" duplicaría preguntas y las insertaría
fuera de orden en la posición alfabética que le toque a "index.txt".
"""

from pathlib import Path


def iter_questions(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            yield q


def source_files(dataset_dir):
    dataset_dir = Path(dataset_dir)
    return sorted(p for p in dataset_dir.glob("*.txt") if p.name != "index.txt")


def load_all_questions(dataset_dir, limit=None):
    questions = []
    for txt in source_files(dataset_dir):
        questions.extend(iter_questions(txt))
    if limit is not None:
        questions = questions[:limit]
    return questions
