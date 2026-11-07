"""Standalone Test Script for Vin Pipeline.

Topic: Commercial Aerospace Economics (Heavy-Lift Launch Vehicles)
This query tests DURALUMIN's ability to handle complex cost-per-kilogram metrics,
payload tonnages, and time-based metrics (turnaround times) across competing entities.
"""

import asyncio
import time
from pathlib import Path
import sys

# Ensure vin is accessible
try:
    import vin
except ImportError:
    print("[ERROR] Could not import vin.py. Make sure this script is in the same directory.")
    sys.exit(1)

async def run_aerospace_test():
    # A highly specific, multi-metric question with zero ambiguity.
    query = (
        "What is the projected cost per kilogram to Low Earth Orbit (LEO) for "
        "SpaceX's fully reusable Starship compared to Blue Origin's New Glenn as of 2025-2026, "
        "and what are their respective maximum payload capacities and estimated turnaround times?"
    )
    
    out_dir = Path("runs/test_aerospace")
    
    print("=" * 80)
    print("🚀 LAUNCHING VIN PIPELINE: COMMERCIAL AEROSPACE ECONOMICS")
    print("=" * 80)
    print(f"QUERY: {query}\n")
    
    # Print the active stage configuration (to confirm Duralumin is loaded)
    vin.print_banner()
    
    t0 = time.time()
    try:
        # Run the full pipeline natively
        dossier_path = await vin.run_pipeline(
            query=query, 
            out_dir=out_dir,
            skip_redteam=False,
            skip_polish=False
        )
        
        elapsed = time.time() - t0
        print("\n" + "=" * 80)
        print(f"[SUCCESS] Vin Pipeline completed successfully in {elapsed:.2f} seconds!")
        print(f"[DOSSIER PATH]: {dossier_path}")
        print("=" * 80)
        
        # Quick summary of output artifacts
        print("\nKey Artifacts Generated:")
        print(f"  1. Duralumin Ledger: {dossier_path.parent / '04_ledger_digest.txt'}")
        print(f"  2. Final Dossier:   {dossier_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Windows-safe asyncio runner
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_aerospace_test())