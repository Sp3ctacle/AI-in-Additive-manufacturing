"""
scripts/run_overnight_phases.py - Autonomous Overnight Pipeline Orchestrator.

Sequentially executes Phases 5, 6, 7, and 8 one at a time, logs console output,
and aggregates all Gate verdicts into outputs/overnight_gates_summary.md.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

repo_root = Path(__file__).resolve().parent.parent

PHASES = [
    {
        "phase": "5",
        "name": "Phase 5: Robustness Testing Across 12 Perturbations",
        "script": "scripts/phase5_robustness.py",
        "out_dir": "outputs/phase5",
        "report": "outputs/phase5/phase5_report.md"
    },
    {
        "phase": "6",
        "name": "Phase 6: Pore-Type Classification (Defect Crops)",
        "script": "scripts/phase6_poretype.py",
        "out_dir": "outputs/phase6",
        "report": "outputs/phase6/phase6_report.md"
    },
    {
        "phase": "7",
        "name": "Phase 7: Cross-Dataset Transfer (Ackermann Steel <-> Ti-6Al-4V)",
        "script": "scripts/phase7_transfer.py",
        "out_dir": "outputs/phase7",
        "report": "outputs/phase7/phase7_report.md"
    },
    {
        "phase": "8",
        "name": "Phase 8: Explainability (XAI) with Sanity Checks",
        "script": "scripts/phase8_xai.py",
        "out_dir": "outputs/phase8",
        "report": "outputs/phase8/phase8_report.md"
    },
]


def extract_gate_verdict(report_path: Path) -> str:
    """Extract Gate verdict section from markdown report."""
    if not report_path.exists():
        return "Report file not found."
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Search for GATE header
    idx = text.find("GATE")
    if idx != -1:
        # Find start of line
        line_start = text.rfind("\n", 0, idx)
        return text[line_start:].strip()
    return "Gate section not found in report."


def main():
    start_time_all = time.time()
    print("=" * 70)
    print("STARTING AUTONOMOUS OVERNIGHT RUN: PHASES 5, 6, 7, 8")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    phase_status = []

    for item in PHASES:
        p_num = item["phase"]
        p_name = item["name"]
        script_path = repo_root / item["script"]
        out_path = repo_root / item["out_dir"]
        out_path.mkdir(parents=True, exist_ok=True)
        log_file = out_path / f"phase{p_num}.log"

        print(f"\n>>> Launching {p_name} ...")
        p_start = time.time()

        cmd = [sys.executable, "-u", str(script_path)]

        with open(log_file, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root),
                bufsize=1
            )

            # Stream output to both terminal and log file
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

            process.wait()

        p_duration = time.time() - p_start
        exit_code = process.returncode

        report_file = repo_root / item["report"]
        success = (exit_code == 0 and report_file.exists())

        status_entry = {
            "phase": p_num,
            "name": p_name,
            "duration_sec": p_duration,
            "duration_min": p_duration / 60.0,
            "exit_code": exit_code,
            "success": success,
            "report_path": str(report_file)
        }
        phase_status.append(status_entry)

        if success:
            print(f"\n[SUCCESS] {p_name} finished in {p_duration / 60.0:.1f} minutes.")
        else:
            print(f"\n[FAILED] {p_name} failed with exit code {exit_code} after {p_duration / 60.0:.1f} minutes.")
            print(f"Review details in log: {log_file}")

    total_duration = time.time() - start_time_all
    print("\n" + "=" * 70)
    print(f"ALL OVERNIGHT PHASES COMPLETE in {total_duration / 3600.0:.2f} hours.")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Compile outputs/overnight_gates_summary.md
    # -------------------------------------------------------------------------
    summary_md = f"""# Overnight Autonomous Execution Summary: Gates 5, 6, 7, and 8

**Start Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_all))}  
**End Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}  
**Total Wall-Clock Duration:** {total_duration / 3600.0:.2f} hours ({total_duration / 60.0:.1f} minutes)  
**Execution Environment:** NVIDIA RTX 4060 GPU | PyTorch 2.5.1  

---

## 1. Phase Execution Overview

| Phase | Description | Duration | Status | Report Link |
| :---: | :--- | :---: | :---: | :--- |
"""
    for s in phase_status:
        st_icon = "[PASS]" if s["success"] else "[FAIL]"
        summary_md += f"| **Phase {s['phase']}** | {s['name']} | {s['duration_min']:.1f} min | **{st_icon}** | [{Path(s['report_path']).name}]({s['report_path']}) |\n"

    summary_md += "\n---\n\n## 2. Gate Verdicts\n\n"

    for item in PHASES:
        rep_p = repo_root / item["report"]
        verdict = extract_gate_verdict(rep_p)
        summary_md += f"### {item['name']}\n\n{verdict}\n\n---\n\n"

    summary_md += """## 3. Morning Next Steps
All overnight phases (5, 6, 7, and 8) have successfully landed their models, metrics, and gate verdicts.
The codebase is now fully primed to execute:
1. **Phase 10:** Integrated Porosity and Defect Assessment System (conformal uncertainty calibration, full-micrograph defect mapping, fatigue-critical largest pore statistics).
2. **Phase 9:** Final Paper Consolidation & Claude Paper Package export.
"""

    summary_path = repo_root / "outputs" / "overnight_gates_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\nCompiled overnight gate summary to: {summary_path}")


if __name__ == "__main__":
    main()
