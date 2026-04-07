"""Check StableText2Brick dataset structure."""
from datasets import load_dataset

ds = load_dataset("AvaLovelace/StableText2Brick")
print(ds)

for s in ds:
    split = ds[s]
    print(f"\n{s}: {len(split)} rows")
    print(f"  columns: {split.column_names}")
    row = split[0]
    for col in split.column_names:
        val = row[col]
        if isinstance(val, str):
            print(f"  {col}: {repr(val[:100])}")
        elif isinstance(val, list):
            print(f"  {col}: list[{len(val)}] first={repr(val[0][:80]) if val and isinstance(val[0], str) else val[0] if val else 'empty'}")
        else:
            print(f"  {col}: {val}")
