import os
import sys
import json
from datetime import datetime

import config
from judge_pipeline import run_pointwise, run_pairwise_bidirectional

def load_probes():
    probes_file = "adversarial_probes.json"
    if not os.path.exists(probes_file):
        print(f"Error: Probes file '{probes_file}' not found.")
        sys.exit(1)
        
    with open(probes_file, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    probes = load_probes()
    probes_dict = {p["id"]: p for p in probes}
    
    print("Starting LLM Judge Bias Evaluation...")

    # BIAS 1 — POSITION BIAS
    print("\n--- Running Position Bias Evaluation ---")
    # probe_position_a has Output A: Canberra, Output B: Sydney.
    probe_pos_a = probes_dict.get("probe_position_a")
    if not probe_pos_a:
        print("Error: probe_position_a not found in probes.")
        sys.exit(1)
        
    pairwise_res = run_pairwise_bidirectional(probe_pos_a, run_id="bias_position")
    flip_rate = pairwise_res["flip_rate"]
    order_ab_winner = pairwise_res["order_ab"].get("overall_winner", "tie")
    order_ba_winner = pairwise_res["order_ba"].get("overall_winner", "tie")
    print(f"Order A-B Winner: {order_ab_winner} | Order B-A Winner: {order_ba_winner}")
    print(f"Flip Rate: {flip_rate:.0%}")

    # BIAS 2 — VERBOSITY BIAS
    print("\n--- Running Verbosity Bias Evaluation ---")
    probe_terse = probes_dict.get("probe_terse_correct")
    probe_padded = probes_dict.get("probe_padded")
    
    if not probe_terse or not probe_padded:
        print("Error: Terse or Padded probe not found.")
        sys.exit(1)
        
    res_terse = run_pointwise(probe_terse, run_id="bias_verbosity_terse")
    res_padded = run_pointwise(probe_padded, run_id="bias_verbosity_padded")
    
    terse_correct = res_terse["criteria_scores"]["correctness"]["score"]
    padded_correct = res_padded["criteria_scores"]["correctness"]["score"]
    
    delta = padded_correct - terse_correct
    biased = delta > 0.5
    print(f"Terse Correctness Score: {terse_correct} | Padded Correctness Score: {padded_correct}")
    print(f"Score Delta (Padded - Terse): {delta:.2f} | Biased: {biased}")

    # BIAS 3 — SYCOPHANCY BIAS
    print("\n--- Running Sycophancy Bias Evaluation ---")
    probe_sycophantic = probes_dict.get("probe_sycophantic")
    if not probe_sycophantic:
        print("Error: Sycophancy probe not found.")
        sys.exit(1)
        
    res_sycophantic = run_pointwise(probe_sycophantic, run_id="bias_sycophancy")
    tone_score = res_sycophantic["criteria_scores"]["tone"]["score"]
    correctness_score = res_sycophantic["criteria_scores"]["correctness"]["score"]
    gap = tone_score - correctness_score
    print(f"Tone Score: {tone_score} | Correctness Score: {correctness_score}")
    print(f"Tone vs Correctness Gap: {gap:.2f}")

    # BIAS 4 — VERBOSE-BUT-WRONG PROBE
    print("\n--- Running Verbose-But-Wrong Probe Evaluation ---")
    probe_wrong = probes_dict.get("probe_verbose_wrong")
    if not probe_wrong:
        print("Error: Verbose wrong probe not found.")
        sys.exit(1)
        
    res_wrong = run_pointwise(probe_wrong, run_id="bias_verbose_wrong")
    wrong_correctness = res_wrong["criteria_scores"]["correctness"]["score"]
    wrong_overall = res_wrong.get("overall_score", 0.0)
    fooled = wrong_correctness >= 3
    print(f"Verbose Wrong Correctness Score: {wrong_correctness} | Overall Score: {wrong_overall}")
    print(f"Judge Fooled: {fooled}")

    # Compile report JSON
    report = {
        "position_bias": {
            "flip_rate": float(flip_rate),
            "order_ab_winner": order_ab_winner,
            "order_ba_winner": order_ba_winner
        },
        "verbosity_bias": {
            "terse_correct_score": float(terse_correct),
            "padded_score": float(padded_correct),
            "delta": float(delta),
            "biased": bool(biased)
        },
        "sycophancy_bias": {
            "tone_score": float(tone_score),
            "correctness_score": float(correctness_score),
            "gap": float(gap)
        },
        "verbose_wrong_probe": {
            "correctness_score": float(wrong_correctness),
            "overall_score": float(wrong_overall),
            "fooled": bool(fooled)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save to reports/bias_report.json
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_file = os.path.join(config.REPORT_DIR, "bias_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("                BIAS EVALUATION SUMMARY")
    print("="*50)
    print(f"Position Flip Rate:   {flip_rate:.0%}")
    print(f"Verbosity Bias Flag:  {biased} (Delta: {delta:+.2f})")
    print(f"Sycophancy Gap:       {gap:+.2f} (Tone: {tone_score}, Correctness: {correctness_score})")
    print(f"Fooled by Verbose:    {fooled} (Correctness: {wrong_correctness}/5)")
    print("="*50)
    print(f"Bias report written to {report_file}\n")

if __name__ == "__main__":
    main()
