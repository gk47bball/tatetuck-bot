"""
evaluate.py — Runner script.
Calls prepare.py's evaluation function with strategy.py's logic.
Outputs the single metric to stdout.

Usage: python evaluate.py > run.log 2>&1
"""
import time
import importlib
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
    print(f"{'Ticker':<8} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    print(f"{'-'*8} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    for r in results["per_ticker"]:
        pred = f"{r['predicted_signal']:+.4f}" if r['predicted_signal'] is not None else "N/A"
        actual = f"{r['actual_signal']:+.4f}" if r['actual_signal'] is not None else "N/A"
        err = f"{r['error']:.4f}" if r['error'] is not None else "N/A"
        print(f"{r['ticker']:<8} {pred:>10} {actual:>10} {err:>10}")
    
    # Print the key metric (same format as Karpathy's val_bpb output)
    print(f"\n---")
    print(f"valuation_error:  {results['valuation_error']:.6f}")
    print(f"num_scored:       {results['num_scored']}")
    print(f"num_total:        {results['num_total']}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    print(f"---")


if __name__ == "__main__":
    main()
