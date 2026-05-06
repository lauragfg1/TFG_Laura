import pandas as pd
import numpy as np

# Leer CSVs
judge_lg = pd.read_csv("03_Langgraph_Parallel/data/8B/heterogeneo/judge_evaluation_results_consistent.csv")
judge_ag = pd.read_csv("04_Autogen_Debate/Resultados/judge_evaluation_results_consistent.csv")

# Filtrar solo Reliable=True
lg_reliable = judge_lg[judge_lg['Reliable'] == True]
ag_reliable = judge_ag[judge_ag['Reliable'] == True]

print("=== LANGGRAPH JUDGE SCORES (RELIABLE ONLY) ===")
print(f"Casos confiables: {len(lg_reliable)}/{len(judge_lg)}")
print(f"Technical_Accuracy: {lg_reliable['Technical_Accuracy'].mean():.2f} ± {lg_reliable['Technical_Accuracy'].std():.2f}")
print(f"Context_Adherence: {lg_reliable['Context_Adherence'].mean():.2f} ± {lg_reliable['Context_Adherence'].std():.2f}")
print(f"Decision_Capability: {lg_reliable['Decision_Capability'].mean():.2f} ± {lg_reliable['Decision_Capability'].std():.2f}")
print(f"Final_Score: {lg_reliable['Final_Score'].mean():.2f} ± {lg_reliable['Final_Score'].std():.2f}")
print(f"Rango: {lg_reliable['Final_Score'].min():.2f} - {lg_reliable['Final_Score'].max():.2f}")

print("\n=== AUTOGEN JUDGE SCORES (RELIABLE ONLY) ===")
print(f"Casos confiables: {len(ag_reliable)}/{len(judge_ag)}")
print(f"Technical_Accuracy: {ag_reliable['Technical_Accuracy'].mean():.2f} ± {ag_reliable['Technical_Accuracy'].std():.2f}")
print(f"Context_Adherence: {ag_reliable['Context_Adherence'].mean():.2f} ± {ag_reliable['Context_Adherence'].std():.2f}")
print(f"Decision_Capability: {ag_reliable['Decision_Capability'].mean():.2f} ± {ag_reliable['Decision_Capability'].std():.2f}")
print(f"Final_Score: {ag_reliable['Final_Score'].mean():.2f} ± {ag_reliable['Final_Score'].std():.2f}")
print(f"Rango: {ag_reliable['Final_Score'].min():.2f} - {ag_reliable['Final_Score'].max():.2f}")

print("\n=== DIFERENCIAS ===")
print(f"Technical_Accuracy: {ag_reliable['Technical_Accuracy'].mean() - lg_reliable['Technical_Accuracy'].mean():+.2f}")
print(f"Context_Adherence: {ag_reliable['Context_Adherence'].mean() - lg_reliable['Context_Adherence'].mean():+.2f}")
print(f"Decision_Capability: {ag_reliable['Decision_Capability'].mean() - lg_reliable['Decision_Capability'].mean():+.2f}")
print(f"Final_Score: {ag_reliable['Final_Score'].mean() - lg_reliable['Final_Score'].mean():+.2f}")
