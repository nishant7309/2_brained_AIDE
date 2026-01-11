
import subprocess
import pandas as pd
import os
import sys
import json
import re

# Fix console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Task settings
TASK_DIR = "aide/example_tasks/house_prices"
GOAL = "Predict house prices"
EVAL = "RMSE"
STEPS = 8

results = []

def run_experiment_cli(name, planner, coder):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {name}")
    print(f"🧠 Planner: {planner}")
    print(f"⚡ Coder:   {coder}")
    print(f"{'='*60}\n")
    
    # Construct CLI command
    cmd = [
        sys.executable, "-m", "aide.run",
        f"data_dir={TASK_DIR}",
        f"goal={GOAL}",
        f"eval={EVAL}",
        f"agent.steps={STEPS}",
        f"agent.planner.model={planner}",
        f"agent.coder.model={coder}",
        "exec.timeout=600"
    ]
    
    # Run and stream output
    # We use check=False so we can parse output even if it 'fails' (e.g. timeout)
    # We let stdout flow to terminal so user sees the UI
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nSkipping to next experiment...")

    # Find the latest log directory to get results
    # Assuming logs are in aide/aideml/logs/ and are sorted by time
    files = sorted(os.scandir("logs"), key=lambda e: e.stat().st_mtime)
    if not files:
        return {"Name": name, "Best RMSE": "N/A", "Planner": planner, "Coder": coder}
        
    latest_log = files[-1].path
    
    # Try to parse best metric from journal or report
    rmse = "N/A"
    try:
        report_path = os.path.join(latest_log, "report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple regex to find RMSE looking numbers
                # This is heuristic; ideally we parse journal.json
                pass 
                
        journal_path = os.path.join(latest_log, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                best_val = None
                for node in data.get('nodes', []):
                    metric = node.get('metric')
                    if metric and isinstance(metric, str) and metric.replace('.','',1).isdigit():
                        val = float(metric)
                        if best_val is None or val < best_val:
                            best_val = val
                    elif metric and isinstance(metric, (int, float)):
                         if best_val is None or metric < best_val:
                            best_val = metric
                
                if best_val is not None:
                    rmse = best_val
                    
    except Exception as e:
        print(f"Error parsing results: {e}")

    return {
        "Name": name,
        "Planner": planner,
        "Coder": coder,
        "Best RMSE": rmse,
        "Log Dir": latest_log
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment.")
        sys.exit(1)

    print("Starting Benchmark...")
    print("Each experiment will generate its own UI.")
    
    try:
        # 1. Run Baseline
        res_baseline = run_experiment_cli(
            "Baseline (All Flash)", 
            planner="gemini-2.0-flash", 
            coder="gemini-2.0-flash"
        )
        results.append(res_baseline)

        # 2. Run Dual-Brain
        res_dual = run_experiment_cli(
            "Dual-Brain (Pro + Flash)", 
            planner="gemini-2.5-pro", 
            coder="gemini-2.0-flash"
        )
        results.append(res_dual)

        # 3. Print Comparison
        print(f"\n{'='*60}")
        print("🏆 FINAL RESULTS COMPARISON")
        print(f"{'='*60}")
        
        df = pd.DataFrame(results)
        print(df[["Name", "Best RMSE", "Planner", "Coder"]].to_string(index=False))
        
        print("\n✅ Comparison Complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark crashed: {e}")
