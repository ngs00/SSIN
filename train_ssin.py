from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score
from util.chem import *
from util.data import load_dataset, collate, load_ref_dataset
from method.model import SSIN


# Set training configurations.
random_seed = 0
num_folds = 5
batch_size = 128
dim_emb = 128
num_epochs = 500
eval_results = list()


for target_fg in func_groups.keys():
    print('------------------------ {} ------------------------'.format(target_fg))

    # Load the IR spectrum dataset.
    # Activate if you need to load a new IR spectrum dataset.
    # dataset = load_dataset(path_metadata='../../data/chem_data/ir/nist/metadata.xlsx',
    #                        path_jdx='../../data/chem_data/ir/nist/jdx',
    #                        idx_smiles=4,
    #                        target_substruct=func_groups[target_fg])
    # torch.save(dataset, 'save/dataset/dataset_{}.pt'.format(target_fg))
    dataset = torch.load('save/dataset/dataset_{}.pt'.format(target_fg))


    # Load the reference dataset.
    dataset_ref = load_ref_dataset(path_metadata='res/metadata_func_group_ref.xlsx',
                                   path_jdx='../../data/chem_data/ir/nist/jdx',
                                   target_substruct=target_fg)

    # Generate sub-datasets for performing k-fold cross validation.
    k_folds = dataset.get_k_folds(num_folds, random_seed=random_seed)
    list_f1 = list()


    # Train and evaluate SSIN based on the k-fold cross validation.
    for k in range(0, num_folds):
        dataset_train = k_folds[k][0]
        dataset_test = k_folds[k][1]

        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)
        target_test = numpy.array([d.label.item() for d in dataset_test.data])
        ids_test = [d.data_id for d in dataset_test.data]

        # Define model, optimizer, and loss function.
        model = SSIN(dim_emb, dataset_train.len_spect, dataset_ref).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-6)
        loss_func = torch.nn.BCEWithLogitsLoss()

        # Optimize model parameters of SSIN.
        for epoch in range(0, num_epochs):
            loss_train = model.fit(loader_train, optimizer, loss_func)
            if (epoch + 1) % 100 == 0:
                print('Fold [{}/{}]\tEpoch [{}/{}]\tTraining loss: {:.3f}'
                      .format(k + 1, num_folds, epoch + 1, num_epochs, loss_train))

        # Execute the trained SSIN on the test dataset.
        pred_test, attns = model.predict(loader_test)
        pred_test = sigmoid(pred_test).numpy().flatten()
        pred_test[pred_test < 0.5] = 0
        pred_test[pred_test >= 0.5] = 1

        # Calculate an evaluation metric on the test dataset.
        list_f1.append(f1_score(target_test, pred_test))

        # Save a model file of the trained SSIN.
        torch.save(model.state_dict(), 'save/model/{}/model_{}.pt'.format(target_fg, k))

    # Print the evaluation results of SSIN on the test datasets.
    print('Test F1-score: {:.3f} ({:.3f})'.format(numpy.mean(list_f1), numpy.std(list_f1)))
