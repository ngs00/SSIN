import pandas
import os
import numpy
import torch
from itertools import chain
from tqdm import tqdm
from rdkit.Chem import MolFromSmarts, MolFromMolFile
from scipy.interpolate import interp1d
from util.chem import get_state_label
from util.data import read_jdx_file


class IRSVecData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.data[idx].x, self.data[idx].y
        else:
            return self.transform(self.data[idx].x.unsqueeze(0).numpy()).squeeze(0), self.data[idx].y

    def set_transform(self, transform):
        self.transform = transform

    def get_k_folds(self, num_folds, random_seed):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.array_split(numpy.random.permutation(len(self.data)), num_folds)
        sub_datasets = list()
        for i in range(0, num_folds):
            sub_datasets.append([self.data[idx] for idx in idx_rand[i]])

        k_folds = list()
        for i in range(0, num_folds):
            dataset_train = IRDataset(list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i+1:])))
            dataset_test = IRDataset(sub_datasets[i])
            k_folds.append([dataset_train, dataset_test])

        return k_folds


def load_dataset(path_metadata, path_jdx, path_mol_file, target_substruct):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    substruct = MolFromSmarts(target_substruct)
    data = list()

    for i in tqdm(range(0, len(metadata))):
        irs = read_jdx_file(path_jdx + '/{}.jdx'.format(metadata[i][0]), norm_y=True, wmin=550, wmax=3801)
        if irs is None:
            continue

        if get_state_label(metadata[i][3]) != 'gas':
            continue

        mol = MolFromMolFile(path_mol_file + '/{}.mol'.format(metadata[i][0]))
        if mol is None:
            continue

        data.append(IRSVecData(torch.tensor(irs.absorbance_savgol, dtype=torch.float),
                               torch.tensor(mol.HasSubstructMatch(substruct), dtype=torch.float)))

    return IRDataset(data)
