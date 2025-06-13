# dataset_lmdb.py
import lmdb, pickle, torch
import numpy as np

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, descriptors, nb_bins):
        self.env = lmdb.open(db_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))

        # Pre-compute bins *once* (requires a scan but not all in RAM)
        self.descriptors = descriptors
        self.nb_bins = nb_bins
        self.bin_values = self._compute_bins()

    # ------------ helpers -----------------------------------------------------
    def _compute_bins(self):
        print('computing bins...')
        # Gather all features for each descriptor across the DB
        all_values = []
        # For each descriptor, collect all values across all samples
        for i, descr in enumerate(self.descriptors):
            data = []
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    if key == b"__len__":
                        continue
                    feats = pickle.loads(value)["feats"]
                    # feats shape: [num_descriptors, latent_size]
                    data.extend(feats[i, :].flatten())
            data = np.array(data, dtype=np.float32)
            data[data < 0] = 0
            data_true = data.copy()
            data.sort()
            index = np.linspace(0, len(data) - 2, self.nb_bins).astype(int)
            values = [data[j] for j in index]
            all_values.append(values)
        print('bins computed')
        return np.array(all_values, dtype=np.float32)

    # ------------ torch Dataset API ------------------------------------------
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(f"{idx:08d}".encode()))
        audio   = torch.tensor(sample["audio"],  dtype=torch.float32)
        feats   = torch.tensor(sample["feats"],  dtype=torch.float32)
        bins    = torch.tensor(self.bin_values, dtype=torch.float32)
        return audio, feats, bins