# build_db.py
import lmdb, pickle, math, numpy as np, torch, librosa as li
from pathlib import Path
from tqdm import tqdm
from config import Config as config                       # existing config
from aux import dummy_load                       # your util
from CLAP import CLAP
import os

AUDIO_DIR   = Path("footsteps_speed_modified")                # raw files
DB_PATH     = "data/footsteps_sm_db"                          # will be ~90 GB per 5 h 44.1 kHz
MAP_SIZE    = 1 << 40                                  # 1 TB – make it big once

target_len  = int(config.AUDIO_LENGTH * config.SAMPLING_RATE)
resampled_len = math.ceil(target_len / (config.N_BAND * np.prod(config.RATIOS)))

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

env = lmdb.open(DB_PATH, map_size=MAP_SIZE)

clap = CLAP(texts=config.TEXTS)

print("computing features...")
with env.begin(write=True) as txn:
    for idx, wav in enumerate(tqdm(list(AUDIO_DIR.rglob("*.wav")))):
        # ─── 1) AUDIO ────────────────────────────────────────────────
        audio = dummy_load(wav, target_len)
        audio = audio.astype(np.float32)               # 1-D numpy
        
        # ─── 2) FEATURES ────────────────────────────────────────────
        feats = clap.get_attribute_score(audio=torch.tensor(audio), 
                descriptors=config.DESCRIPTORS,
                resampled_length = resampled_len)

        sample = {
            "audio": audio,                            # numpy array
            "feats": feats,                            # numpy array
        }
        # key must be bytes; pickle dumps keeps shapes/dtype
        txn.put(key=f"{idx:08d}".encode(), value=pickle.dumps(sample))

    # save total length once (special key)
    txn.put(b"__len__", pickle.dumps(idx + 1))
print("DB written to", DB_PATH)