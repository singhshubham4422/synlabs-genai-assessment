import os
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        # Open in append mode with UTF-8 encoding
        self.file = open(log_path, "a", encoding="utf-8")
        
        self.total_judge_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def log(self, entry: dict):
        """
        Appends a log entry as a single JSON line to the audit file.
        Also tracks cumulative counts.
        """
        # Ensure timestamp is recorded
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat()
            
        json_line = json.dumps(entry, ensure_ascii=False)
        self.file.write(json_line + "\n")
        self.file.flush()

        # Update stats
        self.total_judge_calls += 1
        tokens = entry.get("tokens_used", {})
        self.total_input_tokens += tokens.get("input", 0)
        self.total_output_tokens += tokens.get("output", 0)

    def get_token_summary(self) -> dict:
        """
        Computes total calls, tokens, and estimated cost in USD based on Gemini 3.5 Flash pricing:
        - Input: $1.50 per 1M tokens ($0.0000015/token)
        - Output: $9.00 per 1M tokens ($0.000009/token)
        """
        cost = (self.total_input_tokens * 0.0000015) + (self.total_output_tokens * 0.000009)
        return {
            "total_calls": self.total_judge_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": float(cost)
        }

    def close(self):
        if not self.file.closed:
            self.file.close()
