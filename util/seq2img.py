import pandas
import os
import numpy
import torch
import matplotlib.pyplot as plt
from itertools import chain
from tqdm import tqdm
from scipy.signal import spectrogram
from pywt import cwt
from rdkit.Chem import MolFromSmarts, MolFromInchi, MolFromMolFile
from PIL import Image
from torchvision import transforms
from spkit.cwt import ScalogramCWT
from util.chem import get_state_label
from util.data import read_jdx_file


class IRImgData:
    def __init__(self, img, label):
        self.img = img
        self.label = label


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].img, self.data[idx].label

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


def conv_seq2img(path_metadata, path_ir_spect, path_ir_img, idx_ir_id, method):
    metadata = pandas.read_excel(path_metadata).values.tolist()

    for i in tqdm(range(0, len(metadata))):
        irs = read_jdx_file(path_ir_spect + '/{}.jdx'.format(metadata[i][idx_ir_id]), norm_y=True)
        if irs is None:
            continue

        if method == 'spectrogram':
            f, t, sxx = spectrogram(irs.absorbance)
            plt.figure(figsize=(4, 4))
            plt.pcolormesh(t, f, sxx)
        elif method == 'scalogram':
            coef, freq = cwt(irs.absorbance, numpy.arange(1, 513), 'gaus1')
            plt.figure(figsize=(4, 2))
            plt.imshow(coef, cmap='jet')
        elif method == 'wavelet':
            coef, freq = cwt(irs.absorbance, numpy.arange(1, 513), 'morl')
            plt.figure(figsize=(6, 3))
            plt.imshow(coef, cmap='jet')
        elif method == 'spectrum_image':
            plt.figure(figsize=(8, 2))
            plt.xlim([numpy.min(irs.wavenumber), numpy.max(irs.wavenumber)])
            plt.plot(irs.wavenumber, irs.absorbance, linewidth=1, c='k')

        plt.tight_layout()
        plt.savefig(path_ir_img + '/{}.png'.format(metadata[i][idx_ir_id]))
        plt.close()


def load_dataset(path_metadata, path_irs_img, path_mol_file, target_substruct):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    target_mol = [MolFromSmarts(smt) for _, smt in target_substruct.items()]
    transform = transforms.Compose([transforms.ToTensor()])
    data = list()

    for i in tqdm(range(0, len(metadata))):
        if get_state_label(metadata[i][3]) != 'gas':
            continue

        mol = MolFromMolFile(path_mol_file + '/{}.mol'.format(metadata[i][0]))
        if mol is None:
            continue

        img = Image.open(path_irs_img + '/{}.png'.format(metadata[i][0])).convert('RGB')
        data.append(IRImgData(transform(img),
                              torch.tensor([mol.HasSubstructMatch(tm) for tm in target_mol], dtype=torch.float)))

    return IRDataset(data)
