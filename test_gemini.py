"""
Test AIDE with Gemini dual-brain architecture
"""
import os
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Required for Windows multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    from dotenv import load_dotenv
    load_dotenv()

    import aide

    print("=" * 60)
    print("AIDE Dual-Brain Test with Gemini")
    print("=" * 60)
    print(f"Planner: gemini-2.5-pro (reasoning brain)")
    print(f"Coder: gemini-2.0-flash (execution brain)")
    print("=" * 60)

    # Create experiment
    exp = aide.Experiment(
        data_dir="aide/example_tasks/house_prices",
        goal="Predict house sale prices using the provided training data",
        eval="RMSE (Root Mean Squared Error) - lower is better"
    )

    print("\nStarting experiment with 3 steps...")
    print("This will test the 2-brain architecture:\n")

    # Run for 3 steps to test
    best = exp.run(steps=3)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Best validation score: {best.valid_metric}")
    print(f"Human reviewed: {best.was_human_reviewed}")
    print("\nBest solution code preview:")
    print("-" * 60)
    print(best.code[:1500] + "..." if len(best.code) > 1500 else best.code)
    print("-" * 60)
