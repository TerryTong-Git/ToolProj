import json
from pathlib import Path
import os
import numpy as np

parent_path = Path(__file__).parent

dirs = [d for d in parent_path.iterdir() if d.is_dir()]
file = 'mi_semantics_summary.json'
results_code = []
results_nl = []
for dir in dirs:
    with open(os.path.join(dir, file), 'r') as f:
        obj = json.load(f)
        results_code.append(obj[0]["MI_bits_Z_semantics"])
        results_nl.append(obj[1]["MI_bits_Z_semantics"])
print(f"AVG_MI_CODE: {sum(results_code) / len(results_code):.3g}, STD: {np.array(results_code).std():.3g}")
print(f"AVG_MI_NL: {sum(results_nl) / len(results_nl):.3g}, STD: {np.array(results_nl).std():.3g}")