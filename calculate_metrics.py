import pandas as pd
import numpy as np

# Leer CSVs
langgraph = pd.read_csv("03_Langgraph_Parallel/data/8B/heterogeneo/experiment_metrics.csv")
autogen = pd.read_csv("04_Autogen_Debate/Resultados/metricas_autogen.csv")
judge_lg = pd.read_csv("03_Langgraph_Parallel/data/8B/heterogeneo/judge_evaluation_results_consistent.csv")
judge_ag = pd.read_csv("04_Autogen_Debate/Resultados/judge_evaluation_results_consistent.csv")

print("=== LANGGRAPH METRICS (50 questions) ===")
print(f"Avg_TPS: {langgraph['Avg_TPS'].mean():.2f} ± {langgraph['Avg_TPS'].std():.2f} (range: {langgraph['Avg_TPS'].min():.2f}-{langgraph['Avg_TPS'].max():.2f})")
print(f"Tokens_Out: {langgraph['Tokens_Out'].mean():.0f} ± {langgraph['Tokens_Out'].std():.0f} (range: {langgraph['Tokens_Out'].min():.0f}-{langgraph['Tokens_Out'].max():.0f})")
print(f"Tokens_In: {langgraph['Tokens_In'].mean():.0f} ± {langgraph['Tokens_In'].std():.0f} (range: {langgraph['Tokens_In'].min():.0f}-{langgraph['Tokens_In'].max():.0f})")
print(f"Total_S: {langgraph['Total_S'].mean():.2f} ± {langgraph['Total_S'].std():.2f} (range: {langgraph['Total_S'].min():.2f}-{langgraph['Total_S'].max():.2f})")
print(f"Gen_S: {langgraph['Gen_S'].mean():.2f} ± {langgraph['Gen_S'].std():.2f} (range: {langgraph['Gen_S'].min():.2f}-{langgraph['Gen_S'].max():.2f})")

print("\n=== AUTOGEN METRICS (49 questions) ===")
print(f"Avg_TPS: {autogen['Avg_TPS'].mean():.2f} ± {autogen['Avg_TPS'].std():.2f} (range: {autogen['Avg_TPS'].min():.2f}-{autogen['Avg_TPS'].max():.2f})")
print(f"Tokens_Out: {autogen['Tokens_Out'].mean():.0f} ± {autogen['Tokens_Out'].std():.0f} (range: {autogen['Tokens_Out'].min():.0f}-{autogen['Tokens_Out'].max():.0f})")
print(f"Tokens_In: {autogen['Tokens_In'].mean():.0f} ± {autogen['Tokens_In'].std():.0f} (range: {autogen['Tokens_In'].min():.0f}-{autogen['Tokens_In'].max():.0f})")
print(f"Total_S: {autogen['Total_S'].mean():.2f} ± {autogen['Total_S'].std():.2f} (range: {autogen['Total_S'].min():.2f}-{autogen['Total_S'].max():.2f})")
print(f"Gen_S: {autogen['Gen_S'].mean():.2f} ± {autogen['Gen_S'].std():.2f} (range: {autogen['Gen_S'].min():.2f}-{autogen['Gen_S'].max():.2f})")

print("\n=== LANGGRAPH JUDGE SCORES ===")
print(f"Technical_Accuracy: {judge_lg['Technical_Accuracy'].mean():.2f} ± {judge_lg['Technical_Accuracy'].std():.2f}")
print(f"Context_Adherence: {judge_lg['Context_Adherence'].mean():.2f} ± {judge_lg['Context_Adherence'].std():.2f}")
print(f"Decision_Capability: {judge_lg['Decision_Capability'].mean():.2f} ± {judge_lg['Decision_Capability'].std():.2f}")
print(f"Final_Score: {judge_lg['Final_Score'].mean():.2f} ± {judge_lg['Final_Score'].std():.2f}")

print("\n=== AUTOGEN JUDGE SCORES ===")
print(f"Technical_Accuracy: {judge_ag['Technical_Accuracy'].mean():.2f} ± {judge_ag['Technical_Accuracy'].std():.2f}")
print(f"Context_Adherence: {judge_ag['Context_Adherence'].mean():.2f} ± {judge_ag['Context_Adherence'].std():.2f}")
print(f"Decision_Capability: {judge_ag['Decision_Capability'].mean():.2f} ± {judge_ag['Decision_Capability'].std():.2f}")
print(f"Final_Score: {judge_ag['Final_Score'].mean():.2f} ± {judge_ag['Final_Score'].std():.2f}")

print("\n=== THROUGHPUT COMPARISON ===")
tps_ratio = autogen['Avg_TPS'].mean() / langgraph['Avg_TPS'].mean()
print(f"AutoGen TPS / LangGraph TPS: {tps_ratio:.2f}x (slower by {(1-tps_ratio)*100:.1f}%)")

print("\n=== LATENCY COMPARISON ===")
latency_ratio = autogen['Total_S'].mean() / langgraph['Total_S'].mean()
print(f"AutoGen Total_S / LangGraph Total_S: {latency_ratio:.2f}x (slower by {(latency_ratio-1)*100:.1f}%)")

# Extremos interesantes
print("\n=== EXTREME CASES ===")
print(f"Fastest LangGraph: q{langgraph.loc[langgraph['Avg_TPS'].idxmax(), 'ID']} ({langgraph['Avg_TPS'].max():.2f} TPS)")
print(f"Slowest LangGraph: q{langgraph.loc[langgraph['Avg_TPS'].idxmin(), 'ID']} ({langgraph['Avg_TPS'].min():.2f} TPS)")
print(f"Fastest AutoGen: {autogen.loc[autogen['Avg_TPS'].idxmax(), 'ID']} ({autogen['Avg_TPS'].max():.2f} TPS)")
print(f"Slowest AutoGen: {autogen.loc[autogen['Avg_TPS'].idxmin(), 'ID']} ({autogen['Avg_TPS'].min():.2f} TPS)")
