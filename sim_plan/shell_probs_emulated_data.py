import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

csv_path = Path("data_emulated/emulated_data_output.csv") 
out_path = Path("data_emulated/shell_dist.png")
num_events = 15000000
shells = np.full(num_events, -1, dtype=np.int16)
probs = np.full(num_events, -np.inf, dtype=np.float32)
prob_errs = np.full(num_events, np.nan, dtype=np.float32)
chunksize = 5000000
total_chunks = int(np.ceil(num_events*100/chunksize))

reader = pd.read_csv(csv_path,
        chunksize=chunksize,
        usecols=["theta_query_index", "event_index", "y_cnp", "y_cnp_err"])

for chunk in tqdm(reader, total=total_chunks, unit='chunk'):
            chunk_events = chunk["event_index"].to_numpy(dtype=np.int64) - 1
            chunk_shells = chunk['theta_query_index'].to_numpy(dtype=np.int16) + 1
            chunk_probs = chunk["y_cnp"].to_numpy(dtype=np.float32)
            chunk_prob_errs = chunk["y_cnp_err"].to_numpy(dtype=np.float32)
            
            mask = chunk_probs > probs[chunk_events]

            probs[chunk_events[mask]] = chunk_probs[mask]
            prob_errs[chunk_events[mask]] = chunk_prob_errs[mask]
            shells[chunk_events[mask]] = chunk_shells[mask]

result = pd.DataFrame({
    "event_id": np.arange(num_events),
    "shell": shells,
    "probability": probs,
    "prob_error": prob_errs})

counts = np.bincount(shells, minlength=101)

plt.figure(figsize=(10,5))
plt.bar(np.arange(1, 101), counts[1:101])
plt.xlabel("Shell Number")
plt.ylabel("Count")
plt.title("Distribution of Most Probable Shell")
plt.tight_layout()
plt.savefig(out_path)
