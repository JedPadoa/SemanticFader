# dataset_lmdb.py
import lmdb, pickle, torch
import numpy as np
from config import Config as config

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, descriptors, attribute_name, nb_bins):
        self.db_path = db_path  # Store path instead of environment
        self.descriptors = descriptors
        self.nb_bins = nb_bins
        self.attribute_name = attribute_name
        
        # Open environment temporarily to get length and compute bins
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        
        self.bin_values = self._compute_bins()
        env.close()  # Close the environment
        
        # Don't store the environment as an instance variable
        self.env = None

    def _get_env(self):
        """Get LMDB environment, creating it if necessary (for multiprocessing)"""
        if self.env is None:
            self.env = lmdb.open(self.db_path, readonly=True, lock=False)
        return self.env

    # ------------ helpers -----------------------------------------------------
    def _compute_bins(self):
        print('computing bins...')
        env = lmdb.open(self.db_path, readonly=True, lock=False)
        
        all_values = []
        for i, descr in enumerate(self.descriptors):
            data = []
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key == b"__len__":
                        continue
                    sample = pickle.loads(value)
                    feats = sample[self.attribute_name]
                    
                    data.extend(feats[:].flatten())
                    
            data = np.array(data, dtype=np.float32)
            data[data < 0] = 0
            data_true = data.copy()
            data.sort()
            index = np.linspace(0, len(data) - 2, self.nb_bins).astype(int)
            values = [data[j] for j in index]
            all_values.append(values)
            
        env.close()
        print('bins computed')
        return np.array(all_values, dtype=np.float32)

    # ------------ torch Dataset API ------------------------------------------
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        env = self._get_env()
        with env.begin() as txn:
            sample = pickle.loads(txn.get(f"{idx:08d}".encode()))
        
        audio = torch.tensor(sample["audio"], dtype=torch.float32)
        feats = torch.tensor(sample[self.attribute_name], dtype=torch.float32).unsqueeze(0)
        bins = torch.tensor(self.bin_values, dtype=torch.float32)
        return audio, feats, bins
    
if __name__ == "__main__":
    dataset = AudioDataset(db_path='footsteps_sf_db_speed', descriptors=config.DESCRIPTORS, 
    attribute_name='speed', nb_bins=config.NUM_BINS)
    print('bins shape: ', dataset[0][2].shape)
    print('feats shape: ', dataset[0][1].shape)