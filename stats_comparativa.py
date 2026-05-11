import csv, statistics

def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

lg_m = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\03_Langgraph_Parallel\data\8B\heterogeneo\experiment_metrics.csv")
lg_e = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\03_Langgraph_Parallel\data\8B\heterogeneo\judge_evaluation_results_consistent.csv")

ag_raw = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\04_Autogen_Debate\Resultados\metricas_autogen.csv")
ag_seen = {}
for row in ag_raw:
    ag_seen[row['ID']] = row
ag_m = list(ag_seen.values())
ag_e = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\04_Autogen_Debate\Resultados\judge_evaluation_results_consistent.csv")

cr_m = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\05_CrewAI_Debate\Resultados\metricas_crewai.csv")
cr_e = read_csv(r"c:\Users\Laura\Desktop\TFG_Laura\05_CrewAI_Debate\Resultados\judge_evaluation_results_consistent.csv")

def stats(vals):
    v = [float(x) for x in vals if x not in ('', None)]
    if not v: return 0, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0)

def halluc(rows):
    levels = [r['Hallucination_Level'].strip() for r in rows]
    n = len(levels)
    return n, levels.count('Low'), levels.count('Medium'), levels.count('High')

for name, m, e in [('LangGraph', lg_m, lg_e), ('AutoGen', ag_m, ag_e), ('CrewAI', cr_m, cr_e)]:
    print(f"\n{'='*50}")
    print(f"  {name}  (metrics n={len(m)}, eval n={len(e)})")
    print(f"{'='*50}")
    tps = stats([r['Avg_TPS'] for r in m])
    tot = stats([r['Total_S'] for r in m])
    tout = stats([r['Tokens_Out'] for r in m])
    tin = stats([r['Tokens_In'] for r in m])
    print(f"  TPS:        {tps[0]:.1f} +/- {tps[1]:.2f}")
    print(f"  Total_S:    {tot[0]:.1f} +/- {tot[1]:.1f}")
    print(f"  Tokens_Out: {tout[0]:.0f} +/- {tout[1]:.0f}")
    print(f"  Tokens_In:  {tin[0]:.0f} +/- {tin[1]:.0f}")
    ta = stats([r['Technical_Accuracy'] for r in e])
    ca = stats([r['Context_Adherence'] for r in e])
    dc = stats([r['Decision_Capability'] for r in e])
    fs = stats([r['Final_Score'] for r in e])
    print(f"  Tech Acc:   {ta[0]:.2f} +/- {ta[1]:.2f}")
    print(f"  Ctx Adh:    {ca[0]:.2f} +/- {ca[1]:.2f}")
    print(f"  Dec Cap:    {dc[0]:.2f} +/- {dc[1]:.2f}")
    print(f"  Final:      {fs[0]:.2f} +/- {fs[1]:.2f}")
    n,lo,me,hi = halluc(e)
    print(f"  Halluc Low:    {lo}/{n} ({100*lo//n}%)")
    print(f"  Halluc Medium: {me}/{n} ({100*me//n}%)")
    print(f"  Halluc High:   {hi}/{n} ({100*hi//n}%)")
