"""
evaluate.py — Runner script.
Calls prepare.py's evaluation function with strategy.py's logic.
Outputs the composite metric to stdout.

Usage: python evaluate.py > run.log 2>&1
"""
import time
import prepare
import strategy

def main():
    print("=" * 60)
    print("  BIOPHARMA AUTORESEARCH — EXPERIMENT RUN")
    print("=" * 60)

    start = time.time()

    print("\n[evaluate] Running strategy against benchmark tickers...")
    results = prepare.evaluate_strategy(strategy)

    elapsed = time.time() - start

    # Print per-ticker details
    print(f"\n{'='*60}")
    print(f"{'Ticker':<8} {'Set':<6} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    for r in results["per_ticker"]:
        pred = f"{r['predicted_signal']:+.4f}" if r['predicted_signal'] is not None else "N/A"
        actual = f"{r['actual_signal']:+.4f}" if r['actual_signal'] is not None else "N/A"
        err = f"{r['error']:.4f}" if r['error'] is not None else "N/A"
        s = r.get("set", "?")
        print(f"{r['ticker']:<8} {s:<6} {pred:>10} {actual:>10} {err:>10}")

    # Print the key metrics
    print(f"\n--- TRAIN ({results['num_scored']} scored) ---")
    print(f"valuation_error:  {results['valuation_error']:.6f}")
    print(f"rank_correlation: {results['rank_correlation']:.4f}")
    print(f"dir_accuracy:     {results['directional_accuracy']:.4f}")
    print(f"mae:              {results['mae']:.4f}" if results['mae'] is not None else "mae:              N/A")

    print(f"\n--- HOLDOUT ---")
    print(f"holdout_error:    {results['holdout_valuation_error']:.6f}")
    print(f"holdout_rank:     {results['holdout_rank_correlation']:.4f}")
    print(f"holdout_dir:      {results['holdout_directional_accuracy']:.4f}")
    if results['overfit_warning']:
        print(f"⚠️  OVERFIT WARNING: train-holdout gap = {results['overfit_gap']:.4f}")

    print(f"\n--- SUMMARY ---")
    print(f"num_total:        {results['num_total']}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"---")


if __name__ == "__main__":
    main()
