"""
iTantra Receiver Web App - Benchmark Storage & Export (web_app/benchmark/metrics.py)
===================================================================================
Persists benchmark rows and exports them to JSON and CSV.
"""

import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from web_app.schemas.message import BenchmarkMetrics

class BenchmarkStorage:
    def __init__(self, storage_dir: Path = Path("reports")):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[BenchmarkMetrics] = []

    def record(self, metric: BenchmarkMetrics):
        self.history.append(metric)

    def export_json(self) -> str:
        return json.dumps([m.model_dump() for m in self.history], indent=2)

    def export_csv(self) -> str:
        if not self.history:
            return ""
        keys = list(self.history[0].model_dump().keys())
        lines = [",".join(keys)]
        for m in self.history:
            d = m.model_dump()
            row = [str(d.get(k, "")) for k in keys]
            lines.append(",".join(row))
        return "\n".join(lines)

    def get_comparison(self) -> Dict[str, Any]:
        """Returns comparison between baseline and optimized runs."""
        baselines = [m for m in self.history if m.mode == "BASELINE"]
        optimized = [m for m in self.history if m.mode == "OPTIMIZED"]
        
        def avg(lst, attr):
            vals = [getattr(x, attr) for x in lst if getattr(x, attr, None) is not None]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        return {
            "baseline": {
                "count": len(baselines),
                "avg_ttfa_ms": avg(baselines, "ttfa_ms"),
                "avg_e2e_ms": avg(baselines, "e2e_latency_ms"),
                "avg_rtf": avg(baselines, "rtf"),
                "avg_tts_total_ms": avg(baselines, "tts_total_ms"),
                "avg_peak_ram_mb": avg(baselines, "peak_ram_mb"),
            },
            "optimized": {
                "count": len(optimized),
                "avg_ttfa_ms": avg(optimized, "ttfa_ms"),
                "avg_e2e_ms": avg(optimized, "e2e_latency_ms"),
                "avg_rtf": avg(optimized, "rtf"),
                "avg_tts_total_ms": avg(optimized, "tts_total_ms"),
                "avg_peak_ram_mb": avg(optimized, "peak_ram_mb"),
            }
        }

# Global singleton storage
benchmark_store = BenchmarkStorage()
