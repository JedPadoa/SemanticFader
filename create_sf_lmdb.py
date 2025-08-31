import os
import shutil
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import lmdb
import librosa
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from aux import dummy_load
from config import Config as config
from regCAV import RegCAV

import math
import traceback


def create_sf_lmdb(
    audio_dir: str = ".",
    db_path: str = "footsteps_sf_db",
    attribute_names: list = [],
    rcv_files: list = [],
    resampled_length: int = 100,
    clap_ckpt: str = "laion/clap-htsat-fused"
) -> Optional[str]:
    """
    Create an LMDB database with raw audio and attribute score tensors.
    """
    # 1. Initialize HuggingFace CLAP model & processor
    print(f"Loading Hugging-Face CLAP model: {clap_ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = ClapModel.from_pretrained(clap_ckpt).to(device).eval()
    clap_processor = ClapProcessor.from_pretrained(clap_ckpt)

    all_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    regCav_speed = RegCAV()
    regCav_speed.load_rcv(rcv_files[0])
    regCav_grass = RegCAV()
    regCav_grass.load_rcv(rcv_files[1])
    
    

    # 4. Create LMDB dataset
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print(f"Creating LMDB dataset at: {db_path}")
    map_size = 1 << 40  # 1 TB
    env = lmdb.open(db_path, map_size=map_size)

    # Define audio target length based on config
    target_len = int(config.AUDIO_LENGTH * config.SAMPLING_RATE)

    with env.begin(write=True) as txn:
        sample_count = 0
        for audio_file in tqdm(all_files, desc="Processing files"):
            try:
                # Load raw audio using dummy_load (at 44.1kHz from config)
                raw_audio_44k = dummy_load(f'{audio_dir}/{audio_file}', target_len)
                
                scores = []

                score_speed = regCav_speed.compute_attribute_score(f'{audio_dir}/{audio_file}')
                score_grass = regCav_grass.compute_attribute_score(f'{audio_dir}/{audio_file}')
                
                attribute_score_tensor = torch.stack([
                torch.full((resampled_length,), score_speed, dtype=torch.float32),
                torch.full((resampled_length,), score_grass, dtype=torch.float32)
                ])

                # Store the sample
                sample = {
                    "audio": raw_audio_44k,
                    "attributes": attribute_score_tensor,
                    'file_name': audio_file
                }
                
                key = f"{sample_count:08d}".encode()
                txn.put(key, pickle.dumps(sample))
                sample_count += 1

            except Exception as e:
                error_str = traceback.format_exc()
                print(error_str)
                continue
        
        # Store metadata
        txn.put(b"__len__", pickle.dumps(sample_count))

    print("\n=== SF LMDB Dataset Created ===")
    print(f"Database path: {db_path}")
    print(f"Total samples: {sample_count}")
    return db_path


def main():
    """Main function to create the footstep SF LMDB dataset."""
    print("=== Footstep Semantic Fader (SF) LMDB Dataset Creator ===\n")
    
    # --- Configuration ---
    target_len  = int(config.AUDIO_LENGTH * config.SAMPLING_RATE)
    AUDIO_DIR = "ffxFootstepsGenData"  # Directory containing generated audio files
    DB_PATH = "footsteps_sf_db_speed_grass"      # Path for the output LMDB database
    RCV_FILES = ["RCVs/speed_rcv.pkl", "RCVs/grass_rcv.pkl"]  
    ATTRIBUTE_NAMES = ["speed", "grassiness"]
    RESAMPLED_LENGTH = math.ceil(target_len / (config.N_BAND * np.prod(config.RATIOS)))
    # ---------------------
    
    db_path_created = create_sf_lmdb(AUDIO_DIR, DB_PATH, ATTRIBUTE_NAMES, RCV_FILES, RESAMPLED_LENGTH)
    
    if db_path_created:
        print(f"\nProcessing Complete. Database is ready at: {db_path_created}")
    else:
        print("\nProcessing Failed. Please check the errors above.")


if __name__ == "__main__":
    main()
