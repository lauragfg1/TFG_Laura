from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_summary_row(label: str, metrics_csv: Path, judge_csv: Path) -> dict:
    metrics_df = pd.read_csv(metrics_csv)
    judge_df = pd.read_csv(judge_csv)

    return {
        "Framework": label,
        "Total_S_sum": float(metrics_df["Total_S"].sum()),
        "Total_S_mean": float(metrics_df["Total_S"].mean()),
        "Judge_mean": float(judge_df["Final_Score"].mean()),
        "Judge_std": float(judge_df["Final_Score"].std(ddof=1)),
        "N": int(min(len(metrics_df), len(judge_df))),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    langgraph_metrics = base_dir / "03_Langgraph_Parallel" / "data" / "8B" / "heterogeneo" / "experiment_metrics.csv"
    langgraph_judge = base_dir / "03_Langgraph_Parallel" / "data" / "8B" / "heterogeneo" / "judge_evaluation_results.csv"

    autogen_metrics = base_dir / "04_Autogen_Debate" / "Resultados" / "metricas_autogen.csv"
    autogen_judge = base_dir / "04_Autogen_Debate" / "Resultados" / "judge_evaluation_results.csv"

    output_dir = base_dir / "figuras_memoria"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datasets comparativos...")
    summary_rows = [
        build_summary_row("LangGraph 8B (heter)", langgraph_metrics, langgraph_judge),
        build_summary_row("AutoGen", autogen_metrics, autogen_judge),
    ]
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = output_dir / "langgraph_autogen_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    palette = {
        "LangGraph 8B (heter)": "#1f9d8a",
        "AutoGen": "#d1495b",
    }

    for _, row in summary_df.iterrows():
        x = row["Total_S_sum"]
        y = row["Judge_mean"]
        framework = row["Framework"]

        ax.scatter(
            x,
            y,
            s=270,
            color=palette[framework],
            edgecolor="#1f2937",
            linewidth=1.2,
            zorder=3,
        )

        ax.annotate(
            (
                f"{framework}\n"
                f"Tiempo total: {x:.0f} s\n"
                f"Score juez: {y:.2f}"
            ),
            (x, y),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "#d1d5db",
                "alpha": 0.95,
            },
        )

    ax.axhline(summary_df["Judge_mean"].mean(), color="#9ca3af", linestyle="--", linewidth=1.1, alpha=0.8)

    ax.set_title("Comparativa de rendimiento: LangGraph vs AutoGen", pad=14, fontweight="bold")
    ax.set_xlabel("Tiempo total de ejecucion (s)")
    ax.set_ylabel("Puntuacion media del juez")
    ax.grid(True, alpha=0.25)

    x_min = summary_df["Total_S_sum"].min() * 0.9
    x_max = summary_df["Total_S_sum"].max() * 1.1
    y_min = max(0.0, summary_df["Judge_mean"].min() - 0.6)
    y_max = min(10.0, summary_df["Judge_mean"].max() + 0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    out_png = output_dir / "langgraph_vs_autogen_tiempo_vs_juez.png"
    out_pdf = output_dir / "langgraph_vs_autogen_tiempo_vs_juez.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close(fig)

    print("Figura generada correctamente:")
    print(out_png)
    print(out_pdf)
    print("Resumen numerico:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()