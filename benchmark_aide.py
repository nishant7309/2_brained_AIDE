
import aide
import pandas as pd
import os
import sys

# Fix console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Task settings
TASK_DIR = "aide/example_tasks/house_prices"
GOAL = "Predict house prices"
EVAL = "RMSE"
STEPS = 8  # Enough steps to see divergence

results = []

def run_experiment(name, planner, coder):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {name}")
    print(f"🧠 Planner: {planner}")
    print(f"⚡ Coder:   {coder}")
    print(f"{'='*60}\n")
    
    exp = aide.Experiment(
        data_dir=TASK_DIR,
        goal=GOAL,
        eval=EVAL,
        planner_model=planner,
        coder_model=coder,
    )
    
    # Run the experiment
    best_node = exp.run(steps=STEPS)
    
    # Extract best metric safely
    best_score = best_node.valid_metric if best_node else None
    
    return {
        "Name": name,
        "Planner": planner,
        "Coder": coder,
        "Best RMSE": best_score if best_score is not None else "N/A (Failed)",
        "Steps": len(exp.journal),
        "Log Dir": str(exp.cfg.log_dir)
    }

if __name__ == "__main__":
    # Ensure Windows multiprocessing works
    import multiprocessing
    multiprocessing.freeze_support()

    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found in environment.")
        sys.exit(1)

    print("Starting Benchmark...")
    
    try:
        # 1. Run Baseline (All Flash)
        res_baseline = run_experiment(
            "Baseline (All Flash)", 
            planner="gemini-2.0-flash", 
            coder="gemini-2.0-flash"
        )
        results.append(res_baseline)

        # 2. Run Dual-Brain (Pro + Flash)
        res_dual = run_experiment(
            "Dual-Brain (Pro + Flash)", 
            planner="gemini-2.5-pro", 
            coder="gemini-2.0-flash"
        )
        results.append(res_dual)

        # 3. Print Comparison
        print(f"\n{'='*60}")
        print("🏆 FINAL RESULTS COMPARSION")
        print(f"{'='*60}")
        
        df = pd.DataFrame(results)
        # Handle cases where pd might not display fully
        print(df.to_string(columns=["Name", "Best RMSE", "Planner", "Coder"], index=False))
        
        print("\n✅ Comparison Complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark crashed: {e}")
        import traceback
        traceback.print_exc()
