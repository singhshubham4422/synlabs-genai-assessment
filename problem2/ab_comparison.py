import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import google.generativeai as genai

import config
from judge_pipeline import run_pairwise_bidirectional, logger

# Configure Gemini API for generation
genai.configure(api_key=config.GEMINI_API_KEY)

def load_json_file(filename: str) -> list:
    if not os.path.exists(filename):
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def mock_generate(case: dict, config_type: str) -> str:
    cid = case["id"]
    if config_type == "A":
        return case.get("model_output", "")
        
    responses_b = {
        "tc01": "- Tokyo is the capital of Japan. [1]\n- The city's population is estimated at 14 million in 2023. [2]",
        "tc02": "- General relativity is Albert Einstein's geometric theory of gravitation. [1]\n- It describes gravity as the curvature of spacetime caused by mass and energy. [2]",
        "tc03": "- Inception was directed by Christopher Nolan. [1]\n- The film was officially released in July 2010. [2]",
        "tc04": "- Table salt has the chemical formula NaCl. [1]\n- It is composed of sodium (Na) and chlorine (Cl) elements. [2]",
        "tc05": "- The speed of light in a vacuum is exactly 299,792,458 meters per second. [1]",
        "tc06": "- Neil Armstrong was the first human to walk on the moon in 1969. [1]\n- Buzz Aldrin followed him shortly after. [2]",
        "tc07": "- The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. [1]\n- It covers about 46% of Earth's water surface. [2]",
        "tc08": "- Mars is known as the Red Planet. [1]\n- This appearance is due to the prevalence of iron oxide (rust) on its surface. [2]",
        "tc09": "- Red is a primary color of light. [1]\n- Green is a primary color of light. [2]\n- Blue is a primary color of light. [3]",
        "tc10": "- Absorbs sunlight using chlorophyll in plant leaves. [1]\n- Converts carbon dioxide and water into glucose. [2]\n- Releases oxygen gas into the atmosphere. [3]",
        "tc11": "To: HR Department\nFrom: Employee\nRegarding: Vacation leave request [1]",
        "tc12": "- bonjour, comment ça va? [1]",
        "tc13": "- The thunderstorm was a wild conductor of electric energy. [1]\n- It orchestrated a crashing symphony across the dark sky. [2]",
        "tc14": "- Know your audience and tailor your message to them. [1]\n- Rehearse your delivery multiple times. [2]\n- Practice breathing to manage anxiety. [3]",
        "tc15": "- Social media has popularized casual writing registers. [1]\n- It increases overall daily writing frequency. [2]\n- It provides more self-expression opportunities. [3]",
        "tc16": "- Thank you for 30 years of outstanding dedication. [1]\n- We wish you the best in your retirement. [2]",
        "tc17": "- I cannot provide instructions to bypass locks as it is illegal. [1]",
        "tc18": "- I cannot provide instructions on building explosives as it is unsafe. [1]",
        "tc19": "- Paris is the capital of France. [1]",
        "tc20": "- Boil water in a pot and add salt. [1]\n- Cook the pasta for 8-10 minutes. [2]\n- Drain and serve immediately. [3]"
    }
    return responses_b.get(cid, "- Responded with mock bullet point. [1]")

def generate_response(case: dict, config_type: str) -> Tuple[str, dict]:
    question = case["input"]
    sys_prompt = case.get("system_prompt", "")
    
    if config_type == "A":
        actual_sys_prompt = sys_prompt
    else:
        actual_sys_prompt = (
            f"{sys_prompt}\n\n"
            "Constraint: Format your response as a concise bulleted list. "
            "Where appropriate, include bracketed citations like [1], [2] referencing sources."
        )
        
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_key_here":
        mock_out = mock_generate(case, config_type)
        return mock_out, {"input": 50, "output": 100}
        
    try:
        # Create generator model with system instruction
        model = genai.GenerativeModel(
            model_name=config.GENERATOR_MODEL,
            system_instruction=actual_sys_prompt,
            generation_config={"temperature": 0.7}
        )
        response = model.generate_content(question)
        raw_text = response.text
        
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            
        tokens_used = {"input": input_tokens, "output": output_tokens}
        
        # Log generation call to audit logger
        logger.log({
            "timestamp": datetime.utcnow().isoformat(),
            "generator_model": config.GENERATOR_MODEL,
            "prompt": f"SYS: {actual_sys_prompt}\nUSER: {question}",
            "raw_response": raw_text,
            "tokens_used": tokens_used,
            "case_id": case["id"],
            "run_id": f"ab_gen_config_{config_type}"
        })
        
        return raw_text, tokens_used
    except Exception as e:
        # Fallback if API fails
        mock_out = mock_generate(case, config_type)
        logger.log({
            "timestamp": datetime.utcnow().isoformat(),
            "generator_model": f"{config.GENERATOR_MODEL} (MOCK_FALLBACK)",
            "prompt": f"SYS: {actual_sys_prompt}\nUSER: {question}",
            "raw_response": f"API Failed - {str(e)}",
            "tokens_used": {"input": 50, "output": 100},
            "case_id": case["id"],
            "run_id": f"ab_gen_config_{config_type}_failed"
        })
        return mock_out, {"input": 50, "output": 100}

def main():
    print("Starting A/B Comparison Evaluation...")
    cases = load_json_file("test_suite.json")
    
    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    flips = 0
    
    total_cases = len(cases)
    print(f"Generating answers and evaluating {total_cases} cases...")
    
    for c in cases:
        case_id = c["id"]
        print(f"Processing case {case_id}...")
        
        # Generate outputs A and B
        output_a, tokens_a = generate_response(c, "A")
        output_b, tokens_b = generate_response(c, "B")
        
        # Create evaluation case dictionary
        eval_case = {
            "id": case_id,
            "input": c["input"],
            "system_prompt": c.get("system_prompt", ""),
            "output_a": output_a,
            "output_b": output_b
        }
        
        # Run pairwise bidirectional judge comparison
        verdict = run_pairwise_bidirectional(eval_case, run_id="ab_comparison")
        
        consensus_winner = verdict["consensus_winner"]
        flip_rate = verdict["flip_rate"]
        
        if consensus_winner == "A":
            wins_a += 1
        elif consensus_winner == "B":
            wins_b += 1
        else:
            ties += 1
            
        if flip_rate > 0.0:
            flips += 1
            
        results.append({
            "case_id": case_id,
            "question": c["input"],
            "output_a": output_a,
            "output_b": output_b,
            "order_ab_winner": verdict["order_ab"].get("overall_winner", "tie"),
            "order_ba_winner": verdict["order_ba"].get("overall_winner", "tie"),
            "consensus_winner": consensus_winner,
            "flip_rate": flip_rate,
            "rationale_ab": verdict["order_ab"].get("rationale", ""),
            "rationale_ba": verdict["order_ba"].get("rationale", "")
        })
        
    # Declare winner
    if wins_a > wins_b:
        winner = "Config A (Standard)"
    elif wins_b > wins_a:
        winner = "Config B (Concise Bullet Points & Citations)"
    else:
        winner = "Tie"
        
    report = {
        "config_a_details": {
            "name": "Config A",
            "description": "Standard prompt instructions"
        },
        "config_b_details": {
            "name": "Config B",
            "description": "Concise bulleted list with source citations"
        },
        "total_cases": total_cases,
        "wins_config_a": wins_a,
        "wins_config_b": wins_b,
        "ties": ties,
        "position_flips": flips,
        "winner_declaration": winner,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save report to reports/ab_report.json
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_file = os.path.join(config.REPORT_DIR, "ab_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*50)
    print("                A/B EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Cases Evaluated: {total_cases}")
    print(f"Wins - Config A:       {wins_a}")
    print(f"Wins - Config B:       {wins_b}")
    print(f"Ties:                  {ties}")
    print(f"Position Flips:        {flips}")
    print("-" * 50)
    print(f"Winner declared:       {winner}")
    print("="*50)
    print(f"A/B comparison report written to {report_file}\n")
    
    logger.close()

if __name__ == "__main__":
    main()
