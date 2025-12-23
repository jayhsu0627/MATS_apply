"""
Full Pipeline: Run complete evaluation pipeline.

This script orchestrates all the improvements:
1. Compute vectors for both models
2. Layer stability analysis
3. Orthogonalization
4. Layer ablation
5. LLM-as-Judge evaluation
6. Sanity checks

Run this after generating the dataset.
"""
import os
import sys
import subprocess

def run_step(step_name, script_path, description):
    """Run a pipeline step and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(script_path):
        print(f"⚠️  WARNING: {script_path} not found. Skipping...")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {step_name} failed: {e}")
        return False

def main():
    """Run the full pipeline."""
    print("="*70)
    print("ANALOGY-MAKING STEERING: FULL PIPELINE")
    print("="*70)
    print("\nThis will run all evaluation steps:")
    print("  1. Multi-model vector computation")
    print("  2. Layer stability analysis")
    print("  3. Orthogonalization")
    print("  4. Layer ablation")
    print("  5. LLM-as-Judge evaluation")
    print("  6. Sanity checks")
    print("\nNote: Some steps may take significant time (especially LLM Judge).")
    print("="*70)
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    steps = [
        {
            "name": "Multi-Model Vector Computation",
            "script": "compute_vector_multi.py",
            "description": "Compute analogy vectors for Gemma and Llama models"
        },
        {
            "name": "Layer Stability Analysis",
            "script": "layer_stability_analysis.py",
            "description": "Find where analogy vector direction stabilizes (computation vs propagation)"
        },
        {
            "name": "Orthogonalization",
            "script": "orthogonalize_vector.py",
            "description": "Remove token entanglement (Imagine, like, etc.) from vectors"
        },
        {
            "name": "Layer Ablation",
            "script": "layer_ablation.py",
            "description": "Test steering effectiveness across all layers"
        },
        {
            "name": "Sanity Checks",
            "script": "sanity_checks.py",
            "description": "Random baseline, simplicity confounder, negative steering tests"
        },
        {
            "name": "LLM-as-Judge Evaluation",
            "script": "llm_judge_eval.py",
            "description": "Evaluate steering quality using LLM scoring (may take 30+ min)"
        },
    ]
    
    results = {}
    
    for step in steps:
        success = run_step(step["name"], step["script"], step["description"])
        results[step["name"]] = success
        
        if not success and step["name"] in ["Multi-Model Vector Computation"]:
            print(f"\n⚠️  CRITICAL: {step['name']} failed. Some subsequent steps may not work.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("\nPipeline stopped by user.")
                break
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {step_name}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review generated plots:")
    print("   - layer_stability_*.png")
    print("   - layer_ablation_plot.png")
    print("   - analogy_score_plot_layer*.png (The Money Plot)")
    print("   - sanity_*.png")
    print("\n2. Check results files:")
    print("   - judge_scores_layer*.json")
    print("   - sanity_check_results.json")
    print("\n3. Write executive summary using key insights from:")
    print("   - Layer stability analysis (where computation happens)")
    print("   - Orthogonalization results (token entanglement)")
    print("   - LLM Judge scores (steering effectiveness)")
    print("="*70)

if __name__ == "__main__":
    main()

