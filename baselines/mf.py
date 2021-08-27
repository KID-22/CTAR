import numpy as np
from tqdm import tqdm
import optuna
import random
from optuna.samplers import TPESampler
from optuna.trial import Trial
from utils.progress import WorkSplitter
from scipy.sparse import lil_matrix, csr_matrix
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class Objective:

    def __init__(self, num_users, num_items, train, valid, iters, seed) -> None:
        """Initialize Class"""
        self.num_users = num_users
        self.num_items = num_items
        self.train = train
        self.valid = valid
        self.iters = iters
        self.seed = seed

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        # rank = trial.suggest_discrete_uniform('rank', 4, 64, 4)
        # lam = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        # batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048])
        # lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])

        rank = trial.suggest_categorical('rank', [16])
        lam = trial.suggest_categorical('lambda', [0.1])
        batch_size = trial.suggest_categorical('batch_size', [2048])
        lr = trial.suggest_categorical('learning_rate', [0.005])

        setup_seed(self.seed)

        model = MF(self.num_users, self.num_items, np.int(rank), np.int(batch_size), lamb=lam, learning_rate=lr).cuda()

        score, _, _, _, _ = model.fit(self.train, self.valid, self.iters, self.seed)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, num_users, num_items, train, valid, num_epoch, seed):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(num_users=num_users, num_items=num_items, train=train, valid=valid, iters=num_epoch,
                              seed=seed)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 lamb=0.01,
                 learning_rate=1e-3,
                 **unused):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.lr = learning_rate

        # Variable to learn
        self.user_e = nn.Embedding(self.num_users, self.embed_dim)
        self.item_e = nn.Embedding(self.num_items, self.embed_dim)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.apply(self._init_weights)

        self.loss = nn.MSELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lamb)

    def get_embedding(self):
        return self.user_e, self.item_e, self.user_b, self.item_b

    def fit(self, matrix_train, matrix_valid, num_epoch=100, seed=0):
        setup_seed(seed)

        optimizer = self.get_optimizer()

        # Load data
        ui_pairs = lil_matrix(matrix_train)
        ui_pairs = np.asarray(ui_pairs.nonzero()).T.astype('int32')
        train_label = np.asarray(matrix_train[ui_pairs[:, 0], ui_pairs[:, 1]]).T
        train_label[train_label == -1] = 0

        valid_ui_pairs = lil_matrix(matrix_valid)
        valid_ui_pairs = np.asarray(valid_ui_pairs.nonzero()).T.astype('int32')
        valid_label = np.asarray(matrix_valid[valid_ui_pairs[:, 0], valid_ui_pairs[:, 1]])[0]
        valid_label[valid_label == -1] = 0

        # Training
        train_dataloader = DataLoader(np.hstack((ui_pairs, train_label)), self.batch_size, shuffle=True)
        result, best_result, early_stop, best_U, best_V, best_uB, best_vB = 0, 0, 0, None, None, None, None
        for epoch in tqdm(range(num_epoch)):
            for i, data in enumerate(train_dataloader):
                user = data[:, 0].cuda()
                item = data[:, 1].cuda()
                label = data[:, 2].cuda()

                loss = self.calculate_loss(user.long(), item.long(), label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate
            train_user = torch.LongTensor(ui_pairs[:, 0]).cuda()
            train_item = torch.LongTensor(ui_pairs[:, 1]).cuda()
            train_pred = self.predict(train_user, train_item)
            train_result = roc_auc_score(train_label.flatten(), train_pred.detach().cpu().numpy())

            valid_user = torch.LongTensor(valid_ui_pairs[:, 0]).cuda()
            valid_item = torch.LongTensor(valid_ui_pairs[:, 1]).cuda()
            valid_pred = self.predict(valid_user, valid_item)
            valid_result = roc_auc_score(valid_label, valid_pred.detach().cpu().numpy())

            if valid_result > best_result:
                result = train_result
                best_result = valid_result
                embed_U, embed_V, embed_uB, embed_vB = self.get_embedding()
                best_U, best_V, best_uB, best_vB = embed_U.weight.detach().cpu().numpy(), \
                                                   embed_V.weight.detach().cpu().numpy(), \
                                                   embed_uB.weight.detach().cpu().numpy(), \
                                                   embed_vB.weight.detach().cpu().numpy(),
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > 2:
                    break
        print('training set AUC is {0}'.format(result))
        return best_result, best_U, best_V, best_uB, best_vB


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mf(matrix_train, matrix_valid, matrix_utrain=None, iteration=100, lam=0.01, rank=200, batch_size=500,
       learning_rate=1e-3, seed=0, source=None, searcher='grid', n_trials=1, **unused):
    progress = WorkSplitter()

    progress.section("MF: Set the random seed")
    setup_seed(seed)

    progress.section("MF: Training")
    if source == "unif":  # Source of training data: logged data (None), uniform data ("unif") and both ("combine")
        matrix_train = matrix_utrain

    elif source == "combine":
        ui_pairs = lil_matrix(matrix_train)
        ui_pairs = np.asarray(ui_pairs.nonzero()).T.astype('int32')
        label = np.asarray(matrix_train[ui_pairs[:, 0], ui_pairs[:, 1]]).T

        _ui_pairs = lil_matrix(matrix_utrain)
        _ui_pairs = np.asarray(_ui_pairs.nonzero()).T.astype('int32')
        _label = np.asarray(matrix_utrain[_ui_pairs[:, 0], _ui_pairs[:, 1]]).T

        combine_data = np.hstack((np.vstack((ui_pairs, _ui_pairs)), np.vstack((label, _label))))
        combine_data = np.unique(combine_data, axis=0)

        matrix_train = csr_matrix((combine_data[:, 2], (combine_data[:, 0], combine_data[:, 1])),
                                  shape=matrix_train.shape)

    matrix_input = matrix_train

    m, n = matrix_input.shape

    if searcher == 'optuna':
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, num_users=m, num_items=n, train=matrix_input,
                                         valid=matrix_valid, num_epoch=iteration, seed=seed)
        return trials, best_params

    if searcher == 'grid':
        model = MF(m, n, rank, batch_size, lamb=lam, learning_rate=learning_rate).cuda()

        _, U, V, uB, vB = model.fit(matrix_input, matrix_valid, iteration, seed)

        return U, V, uB, vB
