# simple_download_20k.py
# pip install -U datasets tqdm
from itertools import islice

from datasets import Dataset, load_dataset
from tqdm import tqdm

N = 20_000
# Using a subset with clean streaming support; "default" also works.
# (As of mid-2025, 'default', 'pes2o', and 'wiki' are the reliably loadable splits.)
SUBSET = "wiki"  # or "default" / "pes2o"

it = load_dataset("allenai/olmo-mix-1124", name=SUBSET, split="train", streaming=True, trust_remote_code=True)

rows = [r for r in tqdm(islice(it, N), total=N, desc=f"Taking {N} from {SUBSET}")]
ds = Dataset.from_list(rows)
ds.save_to_disk(f"olmo_20k_{SUBSET}")
print(f"Saved to ./olmo_20k_{SUBSET}")
