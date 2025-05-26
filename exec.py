import os
import pandas
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score
from util.chem import *
from util.data import load_dataset, collate, save_pred_results, save_attns, load_ref_dataset
from method.spect_pm import SpectPM


random_seed = 0
num_folds = 5
batch_size = 128
dim_emb = 128
num_epochs = 500
eval_results = list()


for target_substruct in func_groups.keys():
    print(target_substruct)
    if not os.path.exists('save/model/{}'.format(target_substruct)):
        os.makedirs('save/model/{}'.format(target_substruct))
        os.makedirs('save/pred/{}'.format(target_substruct))
        os.makedirs('save/attn/{}'.format(target_substruct))

    # dataset = load_dataset(path_metadata='../../data/chem_data/ir/nist/metadata.xlsx',
    #                        path_jdx='../../data/chem_data/ir/nist/jdx',
    #                        idx_smiles=4,
    #                        target_substruct=func_groups[target_substruct])
    # torch.save(dataset, 'save/dataset/gas_{}.pt'.format(target_substruct))
    dataset = torch.load('save/dataset/gas_{}.pt'.format(target_substruct))
    k_folds = dataset.get_k_folds(num_folds, random_seed=random_seed)
    print(target_substruct, len(dataset), dataset.num_neg_data, dataset.num_pos_data)

    dataset_ref = load_ref_dataset(path_metadata='res/metadata_func_group_ref.xlsx',
                                   path_jdx='../../data/chem_data/ir/nist/jdx',
                                   target_substruct=target_substruct)
    list_f1 = list()

    for k in range(0, num_folds):
        dataset_train = k_folds[k][0]
        dataset_test = k_folds[k][1]

        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)
        target_test = numpy.array([d.label.item() for d in dataset_test.data])
        ids_test = [d.data_id for d in dataset_test.data]

        model = SpectPM(dim_emb, dataset_train.len_spect, dataset_ref).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-6)
        loss_func = torch.nn.BCEWithLogitsLoss()

        for epoch in range(0, num_epochs):
            loss_train = model.fit(loader_train, optimizer, loss_func)
            pred_test, attns = model.predict(loader_test)
            loss_test = loss_func(pred_test, torch.tensor(target_test, dtype=torch.float).view(-1, 1))

            pred_test = sigmoid(pred_test).numpy().flatten()
            pred_test[pred_test <= 0.5] = 0
            pred_test[pred_test > 0.5] = 1
            f1_test = f1_score(target_test, pred_test)
            if (epoch + 1) % 50 == 0:
                print('{}: Fold [{}/{}]\tEpoch [{}/{}]\tTraining loss: {:.4f}\tTest loss: {:.4f}\tTest F1-score: {:.4f}'
                      .format(target_substruct, k + 1, num_folds, epoch + 1, num_epochs, loss_train, loss_test, f1_test))

        pred_test, attns = model.predict(loader_test)
        pred_test = sigmoid(pred_test).numpy().flatten()
        pred_test[pred_test < 0.5] = 0
        pred_test[pred_test >= 0.5] = 1
        list_f1.append(f1_score(target_test, pred_test))

        torch.save(model.state_dict(), 'save/model/{}/model_{}.pt'.format(target_substruct, k))
        save_pred_results('save/pred/{}/preds_{}.json'.format(target_substruct, k), ids_test, target_test, pred_test)
        save_attns('save/attn/{}/attns_{}.json'.format(target_substruct, k), ids_test, attns.numpy())

    print(target_substruct, numpy.mean(list_f1), numpy.std(list_f1))
    eval_results.append([target_substruct, numpy.mean(list_f1), numpy.std(list_f1)])
    pandas.DataFrame(eval_results).to_csv('eval_results.csv', index=False, header=False)
