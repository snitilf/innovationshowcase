#!/usr/bin/env python3
"""
clear corrupted guardian data before re-collection.
removes:
1. guardian entries from checkpoint (2010-2016)
2. guardian entries from sentiment_scores.csv (2010-2016)
keeps:
- all gdelt data (2017-2023)
- raw articles file (only has gdelt anyway)
"""

import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
checkpoint_file = project_root / "data" / "sentiment" / "fetch_progress.json"
sentiment_file = project_root / "data" / "sentiment" / "sentiment_scores.csv"

print("="*70)
print("CLEARING CORRUPTED GUARDIAN DATA")
print("="*70)

# 1. clear guardian entries from checkpoint
if checkpoint_file.exists():
    with open(checkpoint_file, 'r') as f:
        completed = json.load(f)
    
    completed_set = {(entry[0], entry[1]) for entry in completed}
    
    guardian_entries = {(c, y) for c, y in completed_set if 2010 <= y <= 2016}
    gdelt_entries = {(c, y) for c, y in completed_set if 2017 <= y <= 2023}
    other_entries = {(c, y) for c, y in completed_set if y < 2010 or y > 2023}
    
    print(f"\n[1] Checkpoint:")
    print(f"    Guardian entries to remove: {len(guardian_entries)}")
    print(f"    GDELT entries to keep: {len(gdelt_entries)}")
    
    # keep only gdelt and other entries
    new_checkpoint = gdelt_entries | other_entries
    
    with open(checkpoint_file, 'w') as f:
        json.dump([list(entry) for entry in new_checkpoint], f)
    
    print(f"    ✓ Removed {len(guardian_entries)} guardian entries")
    print(f"    ✓ Kept {len(gdelt_entries)} gdelt entries")

# 2. clear guardian entries from sentiment scores
if sentiment_file.exists():
    df = pd.read_csv(sentiment_file)
    
    guardian_scores = df[df['year'] <= 2016]
    gdelt_scores = df[df['year'] >= 2017]
    
    print(f"\n[2] Sentiment scores:")
    print(f"    Guardian entries to remove: {len(guardian_scores)}")
    if len(guardian_scores) > 0:
        print(f"    Years: {sorted(guardian_scores['year'].unique())}")
        print(f"    Articles: {guardian_scores['article_count'].sum()}")
    
    print(f"    GDELT entries to keep: {len(gdelt_scores)}")
    
    # keep only gdelt scores
    if len(gdelt_scores) > 0:
        gdelt_scores.to_csv(sentiment_file, index=False)
        print(f"    ✓ Removed {len(guardian_scores)} guardian entries")
        print(f"    ✓ Kept {len(gdelt_scores)} gdelt entries")
    else:
        # if no gdelt scores, just remove guardian (file will be recreated)
        if len(guardian_scores) > 0:
            # create empty file with same columns
            empty_df = pd.DataFrame(columns=df.columns)
            empty_df.to_csv(sentiment_file, index=False)
            print(f"    ✓ Removed all guardian entries (file will be recreated)")

print("\n" + "="*70)
print("✓ GUARDIAN DATA CLEARED")
print("="*70)
print("\nNext step: Run Guardian collection")
print("  python3 src/sentiment_analysis.py --provider auto --start-year 2010 --end-year 2016")
print("="*70)

