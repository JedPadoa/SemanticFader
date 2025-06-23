import lmdb
import pickle

class sfLMDB:
    def __init__(self, db_path, attribute_name):
        self.attribute_name = attribute_name
        self.env = lmdb.open(db_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            key = f"{idx:08d}".encode()
            with self.env.begin() as txn:
                sample = pickle.loads(txn.get(key))
            audio = sample["audio"]
            feat = sample[self.attribute_name]
            file_name = sample['file_name']
            return audio, feat, file_name
        else:
            raise TypeError("Index must be an integer.")

    def close(self):
        self.env.close()

if __name__ == "__main__":
    db = sfLMDB("footsteps_sf_db_speed", 'speed')
    audio, speed, file_name = db[333]
    print(audio.shape, speed.shape)
    print(speed[44])
    print(file_name)
    db.close()