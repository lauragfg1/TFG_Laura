"""
Comparativa rapida entre frameworks (LangGraph vs AutoGen vs CrewAI) en la
condicion principal 8B heterogeneo, para revisar de un vistazo los numeros
de las Tablas 5.1/5.2 antes de pasarlos a la memoria.

Lee resumen_metricas_todas_condiciones.csv (generado por
generar_resumen_stats.py) en vez de releer los CSV en bruto: asi esta
comparativa nunca puede desincronizarse de las tablas/figuras, que usan la
misma fuente.

Uso:
    python generar_resumen_stats.py   (primero, si los datos han cambiado)
    python stats_comparativa.py
"""

import csv
from pathlib import Path

RESUMEN_CSV = Path(__file__).resolve().parent / "resumen_metricas_todas_condiciones.csv"

# Condicion de comparacion entre frameworks: 8B heterogeneo en los tres.
OBJETIVO = [
    ("LangGraph", "8B", "heterogeneo"),
    ("AutoGen", "8B", "heterogeneo"),
    ("CrewAI", "8B", "heterogeneo"),
]


def main():
    with open(RESUMEN_CSV, encoding="utf-8") as f:
        rows = {(r["Framework"], r["Modelo"], r["Arquitectura"]): r for r in csv.DictReader(f)}

    for clave in OBJETIVO:
        r = rows.get(clave)
        if r is None:
            print(f"\n⚠️  No hay datos para {clave} en {RESUMEN_CSV.name}."
                  f" Ejecuta antes: python generar_resumen_stats.py")
            continue

        print(f"\n{'='*56}")
        print(f"  {r['Framework']}  (reps={r['Reps_usadas']}, "
              f"debates n={r['N_debates']}, eval total={r['N_eval_total']}, "
              f"fiables={r['N_eval_fiables']})")
        print(f"{'='*56}")

        print(f"  TPS:        {r['TPS_mean']} +/- {r['TPS_std']}")
        print(f"  Total_S:    {r['Total_S_mean']} +/- {r['Total_S_std']}")
        print(f"  Tokens_Out: {r['Tokens_Out_mean']} +/- {r['Tokens_Out_std']}")
        print(f"  Tokens_In:  {r['Tokens_In_mean']} +/- {r['Tokens_In_std']}")

        print(f"  Precisión Técnica:    {r['Tech_Acc_mean']} +/- {r['Tech_Acc_std']}")
        print(f"  Adherencia Contexto:  {r['Ctx_Adh_mean']} +/- {r['Ctx_Adh_std']}")
        print(f"  Capacidad Decisión:   {r['Dec_Cap_mean']} +/- {r['Dec_Cap_std']}")
        print(f"  Puntuación Final:     {r['Final_Score_mean']} +/- {r['Final_Score_std']}")

        print(f"  Alucinación Baja:  {r['Halluc_Low_n']} ({r['Halluc_Low_pct']}%)")
        print(f"  Alucinación Media: {r['Halluc_Med_n']} ({r['Halluc_Med_pct']}%)")
        print(f"  Alucinación Alta:  {r['Halluc_High_n']} ({r['Halluc_High_pct']}%)")


if __name__ == "__main__":
    main()
