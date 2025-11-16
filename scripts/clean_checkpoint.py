#!/usr/bin/env python3
"""clean failed entries from checkpoint."""

import json
from pathlib import Path

checkpoint_file = Path('data/sentiment/fetch_progress.json')
with open(checkpoint_file, 'r') as f:
    completed = json.load(f)

# remove failed venezuela and zimbabwe years (2017-2023 for gdelt)
original_count = len(completed)
completed = [[c, y] for c, y in completed if not (c in ['Venezuela', 'Zimbabwe'] and y >= 2017)]

removed = original_count - len(completed)
print(f'removed {removed} failed entries (venezuela & zimbabwe 2017-2023)')
print(f'remaining entries: {len(completed)}')

with open(checkpoint_file, 'w') as f:
    json.dump(completed, f)

print('checkpoint updated successfully')

