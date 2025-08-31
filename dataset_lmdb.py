import lmdb, pickle, torch
import numpy as np
from config import Config as config

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, descriptors, attribute_name, nb_bins):
        self.db_path = db_path  # Store path instead of environment
        self.descriptors = descriptors
        self.nb_bins = nb_bins
        self.attribute_name = attribute_name
        
        # Open environment temporarily to get length
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        env.close()  # Close the environment

        self.min_val, self.max_val = self._compute_min_max()
        self.bin_values = self._compute_bins()
        
        # Don't store the environment as an instance variable
        self.env = None

    def _get_env(self):
        """Get LMDB environment, creating it if necessary (for multiprocessing)"""
        if self.env is None:
            self.env = lmdb.open(self.db_path, readonly=True, lock=False)
        return self.env

    # ------------ helpers -----------------------------------------------------
    def _compute_min_max(self):
        print('computing min and max...')
        env = lmdb.open(self.db_path, readonly=True, lock=False)
        
        min_val = float('inf')
        max_val = float('-inf')

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key == b"__len__":
                    continue
                sample = pickle.loads(value)
                feats = sample[self.attribute_name]
                
                # Convert to numpy if it's a tensor
                if isinstance(feats, torch.Tensor):
                    feats = feats.numpy()
                
                min_val = min(min_val, np.min(feats))
                max_val = max(max_val, np.max(feats))

        env.close()
        print(f'min: {min_val}, max: {max_val} computed')
        return min_val, max_val

    def _compute_bins(self):
        print('computing bins...')
        env = lmdb.open(self.db_path, readonly=True, lock=False)
        
        all_values = []
        delta = self.max_val - self.min_val

        for i, descr in enumerate(self.descriptors):
            data = []
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key == b"__len__":
                        continue
                    sample = pickle.loads(value)
                    feats = np.array(sample[self.attribute_name][i], dtype=np.float32)

                    if delta == 0:
                        norm_feats = np.zeros_like(feats, dtype=np.float32)
                    else:
                        norm_feats = -1 + (feats - self.min_val) * 2 / delta
                    
                    data.extend(norm_feats.flatten())
                    
            data = np.array(data, dtype=np.float32)
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
        feats_np = np.array(sample[self.attribute_name], dtype=np.float32)
        
        delta = self.max_val - self.min_val
        if delta == 0:
            norm_feats_np = np.zeros_like(feats_np, dtype=np.float32)
        else:
            norm_feats_np = -1 + (feats_np - self.min_val) * 2 / delta
            
        feats = torch.from_numpy(norm_feats_np)
        bins = torch.tensor(self.bin_values, dtype=torch.float32)
        return audio, feats, bins
    
if __name__ == "__main__":
    dataset = AudioDataset(db_path='footsteps_sf_db_speed_grass', descriptors=config.DESCRIPTORS, 
    attribute_name='attributes', nb_bins=config.NUM_BINS)
    print('bins shape: ', dataset[0][2].shape)
    print('feats: ', dataset[4][1])
    print('feats shape: ', dataset[4][1].shape)