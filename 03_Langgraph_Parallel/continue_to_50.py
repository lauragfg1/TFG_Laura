#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cola secuencial: extiende cada combinacion (size, mode, rep) de 30 a 50 preguntas.
Idempotente: comprueba el estado real en disco antes de lanzar cada combo, asi que
se puede interrumpir y volver a ejecutar sin duplicar trabajo.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent.resolve()
PYTHON = HERE.parent / "venv" / "Scripts" / "python.exe"
DATASET = "../Dataset_Preguntas"
OUTPUT = "./data"
LOG = HERE / "continue_to_50.log"

SIZES = ["2B", "8B", "70B"]
DEBATE_MODES = ["heterogeneo", "homogeneo"]
REPS = [1, 2, 3]
TARGET = 50


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def last_question_done(out_dir: Path, n: int) -> bool:
    dec = out_dir / f"q{n:03d}" / f"decision_{n:03d}.json"
    return dec.exists()


def run(args, out_dir: Path):
    log(f"LANZANDO: {' '.join(args)}")
    result = subprocess.run(args, cwd=str(HERE))
    ok = last_question_done(out_dir, TARGET)
    if result.returncode != 0 or not ok:
        log(f"  AVISO: proceso terminado (code={result.returncode}), q{TARGET:03d} presente={ok}")
    else:
        log(f"  OK: q{TARGET:03d} completado.")
    return ok


def main():
    log("=== Iniciando cola de continuacion hasta 50 ===")

    # Debates: heterogeneo / homogeneo
    for size in SIZES:
        for mode in DEBATE_MODES:
            for rep in REPS:
                out_dir = HERE / "data" / size / mode / f"rep{rep}"
                if last_question_done(out_dir, TARGET):
                    log(f"SKIP {size}/{mode}/rep{rep}: ya tiene {TARGET}.")
                    continue
                args = [
                    str(PYTHON), "launch_debate.py", DATASET, OUTPUT,
                    "--size", size, "--mode", mode,
                    "--start", "31", "--limit", str(TARGET),
                    "--rep", str(rep),
                ]
                run(args, out_dir)

    # Individual (baseline single-agente)
    for size in SIZES:
        for rep in REPS:
            out_dir = HERE / "data" / size / "individual" / f"rep{rep}"
            if last_question_done(out_dir, TARGET):
                log(f"SKIP {size}/individual/rep{rep}: ya tiene {TARGET}.")
                continue
            args = [
                str(PYTHON), "launch_single.py", DATASET, OUTPUT,
                "--size", size,
                "--start", "31", "--limit", str(TARGET),
                "--rep", str(rep),
            ]
            run(args, out_dir)

    log("=== Cola de continuacion hasta 50 TERMINADA ===")


if __name__ == "__main__":
    main()
