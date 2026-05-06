#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gpt-oss:120b"
TEMPERATURE = 0.1
NUM_PREDICT = 1536
NUM_CTX = 8192
CONSISTENCY_THRESHOLD = 1.5


CALIBRATION = """
# CALIBRATION EXAMPLES:
- Score 9-10: Response correctly applies Friis equation, cites specific dB margins from context, explains grating lobe suppression mechanism with array factor formula.
- Score 6-8: Correct general principles but missing specific quantitative analysis or partially ignores context.
- Score 3-5: Correct qualitative direction but significant gaps in technical depth or one unverified specific claim.
- Score 0-2: Physically incorrect statements, fabricated metrics or contradicts the reference context directly.
""".strip()


def extract_json(text: str):
    try:
        cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start : end + 1])
    except Exception as exc:
        print(f"⚠️ JSON parse error: {exc}")
    return None


def severity_rank(level: str) -> int:
    levels = {"Low": 0, "Medium": 1, "High": 2}
    return levels.get(str(level).strip().title(), 0)


def severity_label(rank: int) -> str:
    labels = {0: "Low", 1: "Medium", 2: "High"}
    return labels.get(rank, "Low")


def build_prompt(topic: str, consensus_response: str, reference_context: str, inverted: bool = False) -> str:
    rubric = """
# EVALUATION RUBRIC:
Score each criterion from 0.0 to 10.0. Be strict and precise.

1. TECHNICAL_ACCURACY (weight 50%): Are the physics, formulas and engineering principles correct? Penalize heavily any false claim, invented number or physically impossible statement.
2. CONTEXT_ADHERENCE (weight 15%): Does the response correctly use the provided reference context? Penalize if it ignores relevant context or contradicts it.
3. DECISION_CAPABILITY (weight 35%): Does the response arrive at a clear, useful, technically justified decision or conclusion? Does it connect cause and effect with technical precision and actionable direction?

# HALLUCINATION CLASSIFICATION:
- "Low": All claims traceable to context or universal physics/engineering
- "Medium": Contains plausible but unverifiable specific metrics
- "High": Contains fabricated physics, impossible values or dangerous errors

# FINAL SCORE CALCULATION:
base_score = (technical_accuracy * 0.5) + (context_adherence * 0.15) + (decision_capability * 0.35)

Apply hallucination penalty AFTER base score:
- If hallucination_level == "Low": final_score = base_score * 1.0
- If hallucination_level == "Medium": final_score = base_score * 0.75
- If hallucination_level == "High": final_score = base_score * 0.40

Round to one decimal place.

RESPOND ONLY WITH THIS EXACT JSON, NO OTHER TEXT:
{
    "technical_accuracy": 0.0,
    "context_adherence": 0.0,
    "decision_capability": 0.0,
    "hallucination_level": "Low",
    "base_score": 0.0,
    "final_score": 0.0,
    "justification": "Max 80 words explaining the key technical strengths and weaknesses found."
}
""".strip()

    if inverted:
        body = f"""
You are a Senior SATCOM Systems Engineer acting as a Technical Auditor with 20 years of experience in Ka-band link budgets, antenna design and satellite orbital mechanics.

{CALIBRATION}

# CONSENSUS RESPONSE TO EVALUATE:
{consensus_response}

# REFERENCE CONTEXT (extracted from technical SATCOM documentation):
{reference_context}

# ORIGINAL QUESTION:
{topic}

{rubric}
"""
    else:
        body = f"""
You are a Senior SATCOM Systems Engineer acting as a Technical Auditor with 20 years of experience in Ka-band link budgets, antenna design and satellite orbital mechanics.

{CALIBRATION}

# ORIGINAL QUESTION:
{topic}

# REFERENCE CONTEXT (extracted from technical SATCOM documentation):
{reference_context}

# CONSENSUS RESPONSE TO EVALUATE:
{consensus_response}

{rubric}
"""

    return body.strip()


def build_fallback_prompt(topic: str, consensus_response: str, reference_context: str, inverted: bool = False) -> str:
    if inverted:
        return f"""
You are a Senior SATCOM Systems Engineer.

# CONSENSUS RESPONSE TO EVALUATE:
{consensus_response}

# REFERENCE CONTEXT:
{reference_context}

# ORIGINAL QUESTION:
{topic}

Return only valid JSON with these keys:
technical_accuracy, context_adherence, decision_capability, hallucination_level, base_score, final_score, justification.
Keep justification under 80 words.
""".strip()

    return f"""
You are a Senior SATCOM Systems Engineer.

# ORIGINAL QUESTION:
{topic}

# REFERENCE CONTEXT:
{reference_context}

# CONSENSUS RESPONSE TO EVALUATE:
{consensus_response}

Return only valid JSON with these keys:
technical_accuracy, context_adherence, decision_capability, hallucination_level, base_score, final_score, justification.
Keep justification under 80 words.
""".strip()


def hallucination_multiplier(level: str) -> float:
    normalized = str(level).strip().title()
    mapping = {
        "Low": 1.0,
        "Medium": 0.75,
        "High": 0.40,
    }
    return mapping.get(normalized, 0.75)


def normalized_result(raw_data):
    if not raw_data:
        return None

    result = dict(raw_data)

    technical_accuracy = float(result.get("technical_accuracy", 0.0) or 0.0)
    context_adherence = float(result.get("context_adherence", 0.0) or 0.0)
    decision_capability = float(result.get("decision_capability", 0.0) or 0.0)
    hallucination_level = str(result.get("hallucination_level", "Medium")).strip().title()

    base_score = result.get("base_score")
    if base_score is None:
        base_score = (
            technical_accuracy * 0.5
            + context_adherence * 0.15
            + decision_capability * 0.35
        )

    final_score = result.get("final_score")
    if final_score is None:
        final_score = float(base_score) * hallucination_multiplier(hallucination_level)

    result["technical_accuracy"] = round(technical_accuracy, 1)
    result["context_adherence"] = round(context_adherence, 1)
    result["decision_capability"] = round(decision_capability, 1)
    base_score = max(0.0, min(10.0, float(base_score)))
    result["base_score"] = round(base_score, 1)
    final_score = max(0.0, min(10.0, float(final_score)))
    result["final_score"] = round(final_score, 1)
    result["hallucination_level"] = hallucination_level
    result["justification"] = str(result.get("justification", "")).strip()
    return result


def evaluate_once(topic: str, consensus_response: str, reference_context: str, inverted: bool = False):
    prompt = build_prompt(topic, consensus_response, reference_context, inverted=inverted)

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
        "format": "json",
        "keep_alive": -1,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=1200)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = str(message.get("content", ""))
        thinking = str(message.get("thinking", ""))

        # Some reasoning models may exhaust token budget in `thinking` and leave `content` empty.
        candidate_text = content if content.strip() else thinking
        parsed = extract_json(candidate_text)
        if not parsed:
            snippet = candidate_text.replace("\n", " ").strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            print(f"⚠️ Non-JSON judge output ({'inverted' if inverted else 'normal'}): {snippet}")
            fallback_payload = {
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": build_fallback_prompt(topic, consensus_response, reference_context, inverted=inverted)}],
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": 1024,
                    "num_ctx": NUM_CTX,
                },
                "format": "json",
                "keep_alive": -1,
            }

            fallback_response = requests.post(OLLAMA_URL, json=fallback_payload, timeout=1200)
            fallback_response.raise_for_status()
            fallback_data = fallback_response.json()
            fallback_message = fallback_data.get("message", {})
            fallback_content = str(fallback_message.get("content", ""))
            fallback_thinking = str(fallback_message.get("thinking", ""))
            fallback_candidate_text = fallback_content if fallback_content.strip() else fallback_thinking
            fallback_parsed = extract_json(fallback_candidate_text)
            if not fallback_parsed:
                fallback_snippet = fallback_candidate_text.replace("\n", " ").strip()
                if len(fallback_snippet) > 500:
                    fallback_snippet = fallback_snippet[:500] + "..."
                print(f"⚠️ Fallback judge output ({'inverted' if inverted else 'normal'}): {fallback_snippet}")
            return normalized_result(fallback_parsed)

        return normalized_result(parsed)
    except Exception as exc:
        print(f"❌ Connection error with Ollama ({JUDGE_MODEL}): {exc}")
        return None


def merge_hallucination(level_a: str, level_b: str) -> str:
    return severity_label(max(severity_rank(level_a), severity_rank(level_b)))


def evaluate_with_consistency_check(topic: str, consensus_response: str, reference_context: str):
    score1 = evaluate_once(topic, consensus_response, reference_context, inverted=False)
    if not score1:
        return None

    time.sleep(2)

    score2 = evaluate_once(topic, consensus_response, reference_context, inverted=True)
    if not score2:
        score1["consistency_delta"] = None
        score1["reliable"] = False
        return score1

    diff = abs(float(score1["final_score"]) - float(score2["final_score"]))
    avg_score = round((float(score1["final_score"]) + float(score2["final_score"])) / 2.0, 1)

    merged = dict(score1)
    merged["technical_accuracy"] = round((score1["technical_accuracy"] + score2["technical_accuracy"]) / 2.0, 1)
    merged["context_adherence"] = round((score1["context_adherence"] + score2["context_adherence"]) / 2.0, 1)
    merged["decision_capability"] = round((score1["decision_capability"] + score2["decision_capability"]) / 2.0, 1)
    merged["base_score"] = round((score1["base_score"] + score2["base_score"]) / 2.0, 1)
    merged["hallucination_level"] = merge_hallucination(score1.get("hallucination_level"), score2.get("hallucination_level"))
    merged["final_score"] = avg_score
    merged["consistency_delta"] = round(diff, 1)
    merged["reliable"] = diff <= CONSISTENCY_THRESHOLD
    return merged


def find_decision_files(q_folder: Path):
    return list(q_folder.glob("decision_*.json")) + list(q_folder.glob("*_decision.json"))


def main():
    parser = argparse.ArgumentParser(description="Robust LLM-as-a-Judge evaluator with explicit rubric and consistency check")
    parser.add_argument("target_directory", help="Target directory for evaluation (e.g., ./04_Autogen_Debate/Resultados)")
    args = parser.parse_args()

    base_dir = Path(args.target_directory).resolve()
    if not base_dir.exists():
        print(f"❌ The specified directory {base_dir} does not exist.")
        sys.exit(1)

    csv_path = base_dir / "judge_evaluation_results_consistent.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists:
            writer.writerow([
                "Q_ID",
                "Technical_Accuracy",
                "Context_Adherence",
                "Decision_Capability",
                "Hallucination_Level",
                "Base_Score",
                "Final_Score",
                "Consistency_Delta",
                "Reliable",
                "Justification",
            ])

        q_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("q")])

        processed_count = 0
        skipped_already_evaluated = 0
        skipped_missing_decision = 0
        failed_count = 0

        print(f"\n⚖️ Initiating robust evaluation via {JUDGE_MODEL} across {len(q_folders)} directories...\n")

        for q_folder in q_folders:
            q_id = q_folder.name
            decision_files = find_decision_files(q_folder)
            if not decision_files:
                skipped_missing_decision += 1
                continue

            decision_path = decision_files[0]
            evaluation_path = q_folder / f"evaluation_consistent_{q_id[1:]}.json"

            if evaluation_path.exists():
                print(f"⏭️ {q_id} has already been evaluated. Skipping...")
                skipped_already_evaluated += 1
                continue

            with decision_path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    topic = str(data.get("topic", "")).strip()
                    consensus = str(data.get("consensus_response", data.get("response", ""))).strip()
                    reference_context = str(data.get("reference_context", "Context not provided. Evaluate on universal physics.")).strip()

                    topic = topic.replace("\n", " ").replace('"', "'")
                    consensus = consensus.replace("\n", " ").replace('"', "'")
                    reference_context = reference_context.replace("\n", " ").replace('"', "'")
                except Exception as exc:
                    print(f"❌ JSON parsing error encountered in {q_id}: {exc}")
                    failed_count += 1
                    continue

            if not topic or not consensus:
                print(f"⚠️ Skipping {q_id}: Insufficient data (missing topic or response payload).")
                continue

            print(f"👉 Processing evaluation for {q_id}...")
            eval_data = evaluate_with_consistency_check(topic, consensus, reference_context)

            if eval_data:
                with evaluation_path.open("w", encoding="utf-8") as f:
                    json.dump(eval_data, f, indent=4, ensure_ascii=False)

                writer.writerow([
                    q_id,
                    eval_data.get("technical_accuracy", 0.0),
                    eval_data.get("context_adherence", 0.0),
                    eval_data.get("decision_capability", 0.0),
                    eval_data.get("hallucination_level", "Unknown"),
                    eval_data.get("base_score", 0.0),
                    eval_data.get("final_score", 0.0),
                    eval_data.get("consistency_delta", ""),
                    eval_data.get("reliable", False),
                    eval_data.get("justification", ""),
                ])
                f_csv.flush()
                processed_count += 1

                reliability_text = "reliable" if eval_data.get("reliable", False) else "unreliable"
                print(
                    f"   ✅ Judge Score: {eval_data.get('final_score', 0.0)} | "
                    f"Hallucinations: {eval_data.get('hallucination_level', 'Unknown')} | "
                    f"Consistency Δ: {eval_data.get('consistency_delta', 'n/a')} ({reliability_text})"
                )
            else:
                print(f"   ❌ Evaluation failed for {q_id}")
                failed_count += 1

            time.sleep(2)

    print("\n🎉 Evaluation successfully completed!")
    print(
        f"📊 Summary -> Processed: {processed_count} | "
        f"Already evaluated: {skipped_already_evaluated} | "
        f"Missing decision file: {skipped_missing_decision} | "
        f"Failed: {failed_count}"
    )
    print(f"Please review the '{csv_path.name}' file.")


if __name__ == "__main__":
    main()