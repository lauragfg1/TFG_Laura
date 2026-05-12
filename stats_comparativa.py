import csv, statistics

BASE = r"c:\Users\Laura\Desktop\TFG_Laura"

def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def dedup(rows, key='ID'):
    seen = {}
    for r in rows:
        seen[r[key]] = r
    return list(seen.values())

def stats(vals):
    v = [float(x) for x in vals if x not in ('', None)]
    if not v:
        return 0, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0)

def is_reliable(row):
    return row.get('Reliable', '').strip() == 'True'

def halluc(rows):
    levels = [r['Hallucination_Level'].strip() for r in rows]
    n = len(levels)
    return n, levels.count('Low'), levels.count('Medium'), levels.count('High')

lg_m = dedup(read_csv(rf"{BASE}\03_Langgraph_Parallel\data\8B\heterogeneo\experiment_metrics.csv"))
lg_e = dedup(read_csv(rf"{BASE}\03_Langgraph_Parallel\data\8B\heterogeneo\judge_evaluation_results_consistent.csv"), key='Q_ID')

ag_m = dedup(read_csv(rf"{BASE}\04_Autogen_Debate\Resultados\metricas_autogen.csv"))
ag_e = dedup(read_csv(rf"{BASE}\04_Autogen_Debate\Resultados\judge_evaluation_results_consistent.csv"), key='Q_ID')

cr_m = dedup(read_csv(rf"{BASE}\05_CrewAI_Debate\Resultados\metricas_crewai.csv"))
cr_e = dedup(read_csv(rf"{BASE}\05_CrewAI_Debate\Resultados\judge_evaluation_results_consistent.csv"), key='Q_ID')

for name, m, e in [('LangGraph', lg_m, lg_e), ('AutoGen', ag_m, ag_e), ('CrewAI', cr_m, cr_e)]:
    e_rel = [r for r in e if is_reliable(r)]

    print(f"\n{'='*52}")
    print(f"  {name}  (metrics n={len(m)}, eval total={len(e)}, fiables={len(e_rel)})")
    print(f"{'='*52}")

    tps  = stats([r['Avg_TPS']    for r in m])
    tot  = stats([r['Total_S']    for r in m])
    tout = stats([r['Tokens_Out'] for r in m])
    tin  = stats([r['Tokens_In']  for r in m])
    print(f"  TPS:        {tps[0]:.1f} +/- {tps[1]:.2f}")
    print(f"  Total_S:    {tot[0]:.1f} +/- {tot[1]:.1f}")
    print(f"  Tokens_Out: {tout[0]:.0f} +/- {tout[1]:.0f}")
    print(f"  Tokens_In:  {tin[0]:.0f} +/- {tin[1]:.0f}")

    ta = stats([r['Technical_Accuracy']  for r in e_rel])
    ca = stats([r['Context_Adherence']   for r in e_rel])
    dc = stats([r['Decision_Capability'] for r in e_rel])
    fs = stats([r['Final_Score']         for r in e_rel])
    print(f"  Tech Acc:   {ta[0]:.2f} +/- {ta[1]:.2f}  (n={len(e_rel)})")
    print(f"  Ctx Adh:    {ca[0]:.2f} +/- {ca[1]:.2f}")
    print(f"  Dec Cap:    {dc[0]:.2f} +/- {dc[1]:.2f}")
    print(f"  Final:      {fs[0]:.2f} +/- {fs[1]:.2f}")

    n, lo, me, hi = halluc(e)
    print(f"  Halluc Low:    {lo}/{n} ({100*lo//n if n else 0}%)")
    print(f"  Halluc Medium: {me}/{n} ({100*me//n if n else 0}%)")
    print(f"  Halluc High:   {hi}/{n} ({100*hi//n if n else 0}%)")
