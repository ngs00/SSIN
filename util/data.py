import pandas
import numpy
import json
import jcamp
import torch.utils.data
from tqdm import tqdm
from itertools import chain
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from scipy import interpolate
from scipy.signal import savgol_filter
from util.chem import get_state_label


class IRSpectrum:
    def __init__(self, data_id, compound_name, state, wavenumber, absorbance):
        self.data_id = data_id
        self.compound_name = compound_name
        self.state = state
        self.wavenumber = wavenumber
        self.absorbance = absorbance
        self.absorbance_savgol = savgol_filter(absorbance, 32, 3)
        self.label = None

    def to_tensor(self):
        self.wavenumber = torch.tensor(self.wavenumber, dtype=torch.float)
        self.absorbance = torch.tensor(self.absorbance, dtype=torch.float)
        self.absorbance_savgol = torch.tensor(self.absorbance_savgol, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)

    def to_json(self):
        return json.dumps({
            'wavenumber': self.wavenumber.tolist(),
            'absorbance': [numpy.round(val, decimals=3) for val in self.absorbance_savgol],
        })


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.absorb = torch.cat([d.absorbance_savgol.reshape(1, -1) for d in self.data], dim=0).unsqueeze(0).cuda()
        self.absorb.requires_grad = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].absorbance_savgol, self.data[idx].label

    @property
    def len_spect(self):
        return self.data[0].absorbance.shape[0]

    @property
    def num_neg_data(self):
        return sum([d.label.item() == False for d in self.data])

    @property
    def num_pos_data(self):
        return sum([d.label.item() == True for d in self.data])

    def split(self, ratio_train, random_seed):
        num_train = int(ratio_train * len(self.data))

        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.random.permutation(len(self.data))
        dataset_train = IRDataset([self.data[idx] for idx in idx_rand[:num_train]])
        dataset_test = IRDataset([self.data[idx] for idx in idx_rand[num_train:]])

        return dataset_train, dataset_test

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


def load_dataset(path_metadata, path_jdx, idx_smiles, target_substruct):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    substruct = MolFromSmarts(target_substruct)
    data = list()

    for i in tqdm(range(0, len(metadata))):
        irs = read_jdx_file(path_jdx + '/{}.jdx'.format(metadata[i][0]), norm_y=True, wmin=550, wmax=3801)
        if irs is None:
            continue

        if pandas.isnull(metadata[i][idx_smiles]):
            continue

        if get_state_label(metadata[i][3]) != 'gas':
            continue

        mol = MolFromSmiles(metadata[i][idx_smiles])
        if mol is None:
            continue

        irs.label = mol.HasSubstructMatch(substruct)
        irs.to_tensor()
        data.append(irs)

    return IRDataset(data)


def load_ref_dataset(path_metadata, path_jdx, target_substruct):
    metadata = pandas.read_excel(path_metadata, header=None).values.tolist()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        irs = read_jdx_file(path_jdx + '/{}.jdx'.format(metadata[i][0]), norm_y=True, wmin=550, wmax=3801)
        if irs is None:
            continue

        if get_state_label(metadata[i][3]) != 'gas':
            continue

        irs.label = 1 if metadata[i][4] == target_substruct else 0
        irs.to_tensor()
        data.append(irs)

    return IRDataset(data)


def read_jdx_file(file_name, norm_y, wmin=None, wmax=None):
    data_id = file_name.split('/')[-1].split('.jdx')[0]
    spect = jcamp.jcamp_readfile(file_name)

    if spect['yunits'] != 'ABSORBANCE' and spect['yunits'] != 'TRANSMITTANCE':
        return None

    if 'path length' in spect.keys():
        del spect['path length']
    jcamp.jcamp_calc_xsec(spect, skip_nonquant=False)

    if spect['yunits'] == 'ABSORBANCE':
        spect['absorbance'] = spect['y']

    if numpy.min(spect['wavenumbers']) > 1000:
        return None
    if numpy.max(spect['wavenumbers']) < 3000:
        return None

    spect['absorbance'] = numpy.nan_to_num(spect['absorbance'], nan=0)
    spect['wavenumbers'], spect['absorbance'] = interpol_absorbance(spect['wavenumbers'],
                                                                    spect['absorbance'],
                                                                    wmin,
                                                                    wmax)

    if norm_y:
        spect['absorbance'] = spect['absorbance'] / numpy.max(spect['absorbance'])

    state = spect['state'] if 'state' in spect.keys() else 'none'

    return IRSpectrum(data_id, spect['title'], state, spect['wavenumbers'], spect['absorbance'])


def interpol_absorbance(wavenumber, absorbance, wmin, wmax):
    f_interpol = interpolate.interp1d(wavenumber, absorbance, kind='linear', fill_value='extrapolate')

    if wmin is None or wmax is None:
        _wavenumber = numpy.arange(int(numpy.min(wavenumber)), int(numpy.max(wavenumber)), step=2)
    else:
        _wavenumber = numpy.arange(int(wmin), int(wmax), step=2)
    _absorbance = f_interpol(_wavenumber)
    _absorbance = _absorbance.clip(min=0, max=1)

    return _wavenumber, _absorbance


def collate(batch):
    list_absorbance = list()
    list_label = list()

    for b in batch:
        list_absorbance.append(b[0].unsqueeze(0))
        list_label.append(b[1].unsqueeze(0))
    absorbance = torch.cat(list_absorbance, dim=0)
    label = torch.cat(list_label, dim=0)

    return absorbance, label
