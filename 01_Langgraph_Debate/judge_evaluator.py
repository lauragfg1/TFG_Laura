#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import re
import time
import requests
import argparse
from pathlib import Path

# --- CONFIGURACIÓN GLOBAL ---
OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gpt-oss:120b"

# --- ROBUST JSON EXTRACTION ---
def extract_json(text):
    """Safely extracts and parses a JSON block from a larger text response."""
    try:
        # 1. Clean markdown formatting
        text = re.sub(r'```json|```', '', text, flags=re.IGNORECASE).strip()

        # 2. Find the most internal JSON block
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
            
    except Exception as e:
        print(f"⚠️ JSON parse error: {e}")

    return None

# --- 120B JUDGE INVOCATION ---
def evaluate_with_judge(topic, consensus_response, reference_context):
    """Sends the topic and consensus response to the 120B model for quantitative evaluation."""
    
    prompt = f"""You are a Senior SATCOM Systems Engineer acting as a Technical Auditor.
    Your goal is to evaluate an AI-generated consensus.
    
    # ORIGINAL TOPIC/QUESTION:
    {topic}
    
    # REFERENCE CONTEXT:
    {reference_context}

    # CONSENSUS RESPONSE TO EVALUATE:
    {consensus_response}
    
    # EVALUATION RUBRIC (0.0 to 10.0 for final_score):
    1. Evaluate the response and return exactly the following JSON structure.
    2. "hallucination_level": Determine if it's "Low", "Medium", or "High". (Low: Uses context AND/OR correct universal physics. Medium: Claims specific numbers not in context but plausible. High: Fabricated physics or dangerous technical errors).
    3. "final_score": A float from 0.0 to 10.0. The score must be driven by the correctness of the physics/engineering.
       - Consider technical precision and decision capability, but do NOT score them mathematically or average them.
       - Do NOT penalize for adding correct technical facts, physics, or engineering principles that are not explicitly in the Reference Context. 
       - Heavily PENALIZE (drop the score significantly) if there is actual hallucination (false science, fabricated specific metrics/numbers).
    4. "justification": A clear, concise explanation of why this score and hallucination level were given. Max 60 words.

    RESPOND ONLY WITH JSON:
    {{
        "hallucination_level": "Low",
        "final_score": 8.0,
        "justification": "Technical breakdown here..."
    }}
    """

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1, 
            "num_predict": 2048, # High limit to ensure the model finishes its thoughts
            "num_ctx": 4096
        },
        "keep_alive": -1 # Retain the model in memory to optimize subsequent calls
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=1200)
        response.raise_for_status()
        data = response.json()
        content = data['message']['content']
        
        return extract_json(content)
    except Exception as e:
        print(f"❌ Connection error with Ollama ({JUDGE_MODEL}): {e}")
        return None

# --- MAIN EXECUTION LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Quantitative LLM-as-a-Judge Evaluator for Research")
    parser.add_argument("target_directory", help="Target directory for evaluation (e.g., ./data/70B/hetero)")
    args = parser.parse_args()

    base_dir = Path(args.target_directory).resolve()
    if not base_dir.exists():
        print(f"❌ The specified directory {base_dir} does not exist.")
        sys.exit(1)

    # Establish a master CSV file to aggregate all evaluation metrics
    csv_path = base_dir / "judge_evaluation_results.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists:
            writer.writerow(["Q_ID", "Hallucination_Level", "Final_Score", "Justification"])

        # Identify all subdirectories conforming to the query format (e.g., q001, q002)
        q_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("q")])

        processed_count = 0
        skipped_already_evaluated = 0
        skipped_missing_decision = 0
        failed_count = 0
        
        print(f"\n⚖️ Initiating rigorous evaluation via {JUDGE_MODEL} across {len(q_folders)} directories...\n")

        for q_folder in q_folders:
            q_id = q_folder.name 
            
            # Locate the decision output file (supports both naming conventions)
            decision_files = list(q_folder.glob("decision_*.json")) + list(q_folder.glob("*_decision.json"))
            if not decision_files:
                skipped_missing_decision += 1
                continue
                
            decision_path = decision_files[0]
            evaluation_path = q_folder / f"evaluation_{q_id[1:]}.json" 
            
            # Bypassing previously evaluated queries to ensure resume capability
            if evaluation_path.exists():
                print(f"⏭️ {q_id} has already been evaluated. Skipping...")
                skipped_already_evaluated += 1
                continue

            # Read the input payload generated during the AI debate
            with decision_path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    topic = str(data.get("topic", "")).strip()
                    # Evaluamos consensus_response (LangGraph Debate) o response (LangGraph Single)
                    consensus = str(data.get("consensus_response", data.get("response", ""))).strip()
                    reference_context = str(data.get("reference_context", "Context not provided. Evaluate on universal physics.")).strip()

                    # Clean newlines and quotes to prevent prompt injection errors
                    topic = topic.replace("\n", " ").replace('"', "'")
                    consensus = consensus.replace("\n", " ").replace('"', "'")
                    reference_context = reference_context.replace("\n", " ").replace('"', "'")

                except Exception as e:
                    print(f"❌ JSON parsing error encountered in {q_id}: {e}")
                    continue
            
            if not topic or not consensus:
                print(f"⚠️ Skipping {q_id}: Insufficient data (missing topic or response payload).")
                continue
                
            print(f"👉 Processing evaluation for {q_id}...")
            
            # Execute the LLM evaluation
            eval_data = evaluate_with_judge(topic, consensus, reference_context)
            
            if eval_data:
                # Safely extract metrics, formatting correctly
                hallucination_level = eval_data.get("hallucination_level", "Unknown")
                try:
                    llm_score = float(eval_data.get("final_score", 0))
                except:
                    llm_score = 0.0
                justification = eval_data.get("justification", "")
                
                # Write individual evaluation to a dedicated JSON file
                with evaluation_path.open("w", encoding="utf-8") as f:
                    json.dump(eval_data, f, indent=4)
                
                # Append metrics to the master CSV
                writer.writerow([
                    q_id,
                    hallucination_level,
                    llm_score,              
                    justification
                ])
                f_csv.flush() # Force disk write
                processed_count += 1
                
                print(f"   ✅ Judge Score: {llm_score} | Hallucinations: {hallucination_level}")
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
    print("Please review the 'judge_evaluation_results.csv' file.")

if __name__ == "__main__":
    main()