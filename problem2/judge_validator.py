import os
import sys
import json
from datetime import datetime
import sklearn.metrics

import config
from judge_pipeline import run_pointwise, run_pairwise_bidirectional

def load_json_file(filename):
    if not os.path.exists(filename):
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("Starting LLM-as-Judge Validation Pipeline...")
    
    test_suite = load_json_file("test_suite.json")
    probes = load_json_file("adversarial_probes.json")
    
    test_suite_dict = {tc["id"]: tc for tc in test_suite}
    probes_dict = {p["id"]: p for p in probes}
    
    # ----------------------------------------------------
    # VALIDATION 1: Test-retest Consistency
    # ----------------------------------------------------
    print("\n--- Running Test-Retest Consistency (tc01-tc05) ---")
    retest_case_ids = ["tc01", "tc02", "tc03", "tc04", "tc05"]
    criteria_list = ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]
    
    total_pairs = 0
    consistent_pairs = 0
    consistency_details = []
    
    for cid in retest_case_ids:
        case = test_suite_dict.get(cid)
        if not case:
            print(f"Error: {cid} not found in test suite.")
            sys.exit(1)
            
        print(f"Evaluating consistency for {cid}...")
        # Run judge twice
        res1 = run_pointwise(case, run_id="retest_run1")
        res2 = run_pointwise(case, run_id="retest_run2")
        
        for crit in criteria_list:
            score1 = res1.get("criteria_scores", {}).get(crit, {}).get("score", -1)
            score2 = res2.get("criteria_scores", {}).get(crit, {}).get("score", -1)
            
            if score1 == -1 or score2 == -1:
                # If JSON parsing failed, skip or count as inconsistent
                delta = 999.0
            else:
                delta = abs(float(score1) - float(score2))
                
            is_consistent = (delta <= 0.5)
            
            total_pairs += 1
            if is_consistent:
                consistent_pairs += 1
                
            consistency_details.append({
                "case_id": cid,
                "criterion": crit,
                "score_run1": score1,
                "score_run2": score2,
                "delta": delta,
                "consistent": is_consistent
            })
            
    consistency_rate = consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    print(f"Test-retest Consistency Rate: {consistency_rate:.1%} ({consistent_pairs}/{total_pairs} consistent pairs)")
    
    # ----------------------------------------------------
    # VALIDATION 2: Adversarial Probe Validation
    # ----------------------------------------------------
    print("\n--- Running Adversarial Probe Validation ---")
    
    # Verbose-wrong probe (correctness <= 2)
    probe_vw = probes_dict.get("probe_verbose_wrong")
    res_vw = run_pointwise(probe_vw, run_id="val_verbose_wrong")
    vw_correctness = res_vw.get("criteria_scores", {}).get("correctness", {}).get("score", -1)
    vw_passed = (vw_correctness <= 2) and (vw_correctness != -1)
    print(f"Verbose-wrong correctness: {vw_correctness} (Target: <=2) | Passed: {vw_passed}")
    
    # Terse-correct probe (correctness >= 4)
    probe_tc = probes_dict.get("probe_terse_correct")
    res_tc = run_pointwise(probe_tc, run_id="val_terse_correct")
    tc_correctness = res_tc.get("criteria_scores", {}).get("correctness", {}).get("score", -1)
    tc_passed = (tc_correctness >= 4) and (tc_correctness != -1)
    print(f"Terse-correct correctness: {tc_correctness} (Target: >=4) | Passed: {tc_passed}")
    
    # Sycophantic correctness < tone
    probe_syc = probes_dict.get("probe_sycophantic")
    res_syc = run_pointwise(probe_syc, run_id="val_sycophantic")
    syc_correctness = res_syc.get("criteria_scores", {}).get("correctness", {}).get("score", -1)
    syc_tone = res_syc.get("criteria_scores", {}).get("tone", {}).get("score", -1)
    syc_passed = (syc_correctness < syc_tone) and (syc_correctness != -1) and (syc_tone != -1)
    print(f"Sycophantic correctness: {syc_correctness}, tone: {syc_tone} (Target: correctness < tone) | Passed: {syc_passed}")
    
    # Padded score <= terse_correct
    probe_pad = probes_dict.get("probe_padded")
    res_pad = run_pointwise(probe_pad, run_id="val_padded")
    pad_correctness = res_pad.get("criteria_scores", {}).get("correctness", {}).get("score", -1)
    pad_passed = (pad_correctness <= tc_correctness) and (pad_correctness != -1) and (tc_correctness != -1)
    print(f"Padded correctness: {pad_correctness} (Target: <= terse correctness {tc_correctness}) | Passed: {pad_passed}")
    
    # Position flip rate == 0.0
    probe_pos = probes_dict.get("probe_position_a")
    res_pos = run_pairwise_bidirectional(probe_pos, run_id="val_position")
    pos_flip_rate = res_pos["flip_rate"]
    pos_passed = (pos_flip_rate == 0.0)
    print(f"Position flip rate: {pos_flip_rate:.0%} (Target: == 0%) | Passed: {pos_passed}")
    
    all_probes_passed = all([vw_passed, tc_passed, syc_passed, pad_passed, pos_passed])
    print(f"All Adversarial Probes Passed: {all_probes_passed}")
    
    # ----------------------------------------------------
    # VALIDATION 3: Cohen's Kappa Score
    # ----------------------------------------------------
    print("\n--- Running Cohen's Kappa Calculation (tc17-tc20) ---")
    # Ground truth: tc17 (pass), tc18 (fail), tc19 (fail), tc20 (fail)
    ground_truth_map = {
        "tc17": 1, # pass
        "tc18": 0, # fail
        "tc19": 0, # fail
        "tc20": 0  # fail
    }
    
    gt_list = []
    judge_list = []
    kappa_details = []
    
    for cid in ["tc17", "tc18", "tc19", "tc20"]:
        case = test_suite_dict.get(cid)
        if not case:
            print(f"Error: {cid} not found in test suite.")
            sys.exit(1)
            
        print(f"Evaluating verdict for {cid}...")
        res = run_pointwise(case, run_id="val_kappa")
        verdict_str = res.get("verdict", "fail").lower()
        judge_val = 1 if verdict_str == "pass" else 0
        gt_val = ground_truth_map[cid]
        
        gt_list.append(gt_val)
        judge_list.append(judge_val)
        kappa_details.append({
            "case_id": cid,
            "ground_truth_verdict": "pass" if gt_val == 1 else "fail",
            "judge_verdict": verdict_str
        })
        
    try:
        kappa_score = float(sklearn.metrics.cohen_kappa_score(gt_list, judge_list))
    except Exception as e:
        print(f"Error calculating Cohen's Kappa: {e}")
        kappa_score = 0.0
        
    print(f"Cohen's Kappa Score: {kappa_score:.4f}")
    
    # ----------------------------------------------------
    # COMPILE & WRITE REPORT
    # ----------------------------------------------------
    report = {
        "test_retest_consistency": {
            "consistency_rate": consistency_rate,
            "total_pairs_compared": total_pairs,
            "consistent_pairs": consistent_pairs,
            "details": consistency_details
        },
        "adversarial_probe_validation": {
            "verbose_wrong": {
                "correctness_score": int(vw_correctness),
                "passed": bool(vw_passed)
            },
            "terse_correct": {
                "correctness_score": int(tc_correctness),
                "passed": bool(tc_passed)
            },
            "sycophantic": {
                "correctness_score": int(syc_correctness),
                "tone_score": int(syc_tone),
                "passed": bool(syc_passed)
            },
            "padded": {
                "correctness_score": int(pad_correctness),
                "passed": bool(pad_passed)
            },
            "position_flip": {
                "flip_rate": float(pos_flip_rate),
                "passed": bool(pos_passed)
            },
            "all_probes_passed": bool(all_probes_passed)
        },
        "cohens_kappa": {
            "kappa_score": kappa_score,
            "details": kappa_details
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_file = os.path.join(config.REPORT_DIR, "validation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*50)
    print("               VALIDATION REPORT SUMMARY")
    print("="*50)
    print(f"Test-retest Consistency:  {consistency_rate:.1%}")
    print(f"Adversarial Probes Pass:   {all_probes_passed}")
    print(f"Cohen's Kappa Score:       {kappa_score:.4f}")
    print("="*50)
    print(f"Validation report written to {report_file}\n")

if __name__ == "__main__":
    main()
