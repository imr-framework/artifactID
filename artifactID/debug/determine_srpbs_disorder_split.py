from pathlib import Path

import numpy as np
import pandas as pd

path_inference = Path(r"D:\MMJ\Data\SRPBS-T1-selection")
folders = list(path_inference.glob("*"))
# Get unique sub numbers
sub_inference = list(map(lambda item: int(item.name.split("-")[1]), folders))
sub_inference = np.unique(sub_inference)

path_participants = Path(r"D:\MMJ\Data\SRPBS-T1\participants.tsv")
participants = pd.read_csv(str(path_participants), sep='\t')
participants = participants[["participant_id", "diag"]]  # Filter columns
participants["participant_id"] = participants["participant_id"].apply(
    lambda item: int(item.split("-")[1]))  # Fix subject names
index = ((participants[
    participants["participant_id"].isin(sub_inference)]).index)  # Indices of subjects that are used in our work
participants = participants.loc[index]  # Filter by these indices
print(participants)
print("---")
print(participants.diag.value_counts())