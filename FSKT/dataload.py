import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import get_num_workers


def get_group(df):
    return df[
        ['uid', 'questions', 'responses', 'timestamps', 'usetimes']
    ].groupby('uid').apply(lambda r: (
        r['uid'].values,
        r['questions'].values,      
        r['responses'].values,      
        r['timestamps'].values,     
        r['usetimes'].values,       
    ))


class KTDataset(Dataset):
    def __init__(self, group, max_seq, min_seq=2):
        self.max_seq = max_seq
        self.min_seq = min_seq
        self.samples = {}
        self.user_ids = []

        for user_id in group.index:
            uid, qseqs, rseqs, tseqs, utseqs = group[user_id]
            seq_len = len(qseqs)

            if seq_len < min_seq:
                continue

            if seq_len > max_seq:
                initial = seq_len % max_seq
                if initial >= min_seq:
                    sample_id = f"{user_id}_0"
                    self.user_ids.append(sample_id)
                    self.samples[sample_id] = (
                        uid[:initial], qseqs[:initial],
                        rseqs[:initial], tseqs[:initial], utseqs[:initial]
                    )

                num_full = seq_len // max_seq
                for seq in range(num_full):
                    sample_id = f"{user_id}_{seq + 1}"
                    self.user_ids.append(sample_id)
                    start = initial + seq * max_seq
                    end = initial + (seq + 1) * max_seq
                    self.samples[sample_id] = (
                        uid[start:end], qseqs[start:end],
                        rseqs[start:end], tseqs[start:end], utseqs[start:end]
                    )
            else:
                sample_id = str(user_id)
                self.user_ids.append(sample_id)
                self.samples[sample_id] = (uid, qseqs, rseqs, tseqs, utseqs)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        uid_, qseqs_, rseqs_, tseqs_, utseqs_ = self.samples[user_id]
        seq_len = len(qseqs_)

        uid = np.zeros(self.max_seq, dtype=int)
        qseqs = np.zeros(self.max_seq, dtype=int)
        rseqs = np.zeros(self.max_seq, dtype=int)
        tseqs = np.zeros(self.max_seq, dtype=int)
        utseqs = np.zeros(self.max_seq, dtype=int)
        smasks = np.zeros(self.max_seq, dtype=int)

        uid[:seq_len] = uid_
        qseqs[:seq_len] = qseqs_
        rseqs[:seq_len] = rseqs_
        tseqs[:seq_len] = tseqs_
        utseqs[:seq_len] = utseqs_
        smasks[:seq_len] = 1

        return uid, qseqs, rseqs, tseqs, utseqs, smasks


def load_dataset(opt):
    paths = opt.paths

    if not os.path.exists(paths['data']):
        raise FileNotFoundError(
            f"Data file does not exist: {paths['data']}\n"
            f"Please run: python data/process_data.py --dataset {opt.dataset}"
        )

    data = pd.read_csv(paths['data'])
    print(f"dataset: {opt.dataset} | {len(data)} Interactions | {data['uid'].nunique()} users")
    return data


def load_q_matrix(opt):
    q_matrix_path = opt.paths.get('q_matrix')
    if q_matrix_path and os.path.exists(q_matrix_path):
        Q = np.load(q_matrix_path)
        print(f"Q Matrix: {Q.shape}")
        return Q
    else:
        raise FileNotFoundError(f"The Q matrix does not exist: {q_matrix_path}")


def load_split(path, cv=None):
    if cv is not None:
        path = path.format(cv=cv)
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_dataloader(df, opt, shuffle=False):
    group = get_group(df)
    dataset = KTDataset(group, max_seq=opt.length, min_seq=2)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=get_num_workers(),
        shuffle=shuffle
    )
