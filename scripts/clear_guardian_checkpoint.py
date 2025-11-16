#!/usr/bin/env python3
"""
selectively clear guardian entries (2010-2016) from checkpoint
so guardian data can be re-collected without affecting gdelt entries.
"""

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
checkpoint_file = project_root / "data" / "sentiment" / "fetch_progress.json"

if not checkpoint_file.exists():
    print("⚠ checkpoint file not found")
    exit(1)

# load checkpoint
with open(checkpoint_file, 'r') as f:
    completed = json.load(f)

# convert to set of tuples
completed_set = {(entry[0], entry[1]) for entry in completed}

# separate guardian and gdelt entries
guardian_entries = {(c, y) for c, y in completed_set if 2010 <= y <= 2016}
gdelt_entries = {(c, y) for c, y in completed_set if 2017 <= y <= 2023}
other_entries = {(c, y) for c, y in completed_set if y < 2010 or y > 2023}

print(f"current checkpoint:")
print(f"  guardian (2010-2016): {len(guardian_entries)} entries")
print(f"  gdelt (2017-2023):    {len(gdelt_entries)} entries")
if other_entries:
    print(f"  other years:          {len(other_entries)} entries")

# keep only gdelt entries (remove guardian)
new_checkpoint = gdelt_entries | other_entries

print(f"\nafter clearing guardian entries:")
print(f"  remaining entries: {len(new_checkpoint)}")

# save updated checkpoint
with open(checkpoint_file, 'w') as f:
    json.dump([list(entry) for entry in new_checkpoint], f)

print(f"\n✓ cleared {len(guardian_entries)} guardian entries from checkpoint")
print(f"  kept {len(gdelt_entries)} gdelt entries")
print(f"  guardian data can now be re-collected")

