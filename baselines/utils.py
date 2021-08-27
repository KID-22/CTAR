from torch.utils import data
import numpy as np
from metrics import AUC, MAE, MSE, RMSE, MAE_ips, MSE_ips, RMSE_ips
import torch


class MF_DATA(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename, dtype=int, delimiter=',', skiprows=1)
        self.users_num = int(1000)
        self.items_num = int(1720)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class CausE_DATA(data.Dataset):
    def __init__(self, s_c_data, s_t_data):
        raw_matrix_c = np.loadtxt(s_c_data, dtype=int, delimiter=',', skiprows=1)
        raw_matrix_t = np.loadtxt(s_t_data, dtype=int, delimiter=',', skiprows=1)
        self.s_c = raw_matrix_c
        self.s_t = raw_matrix_t
        raw_matrix = np.vstack((raw_matrix_c, raw_matrix_t))
        self.users_num = int(1000)
        self.items_num = int(1720)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def evaluate_model(model, val_data, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    if np.count_nonzero(true == 1) == true.shape[0] or np.count_nonzero(true == 0) == true.shape[0]:
        auc = 0
    else:
        auc = AUC(true, preds.detach().cpu().numpy())

    return mae, mse, rmse, auc


def evaluate_IPS_model(model, val_data, inverse_propensity, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    user_num = max(user)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    item_num = max(item)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE_ips(preds, true, item, user_num, item_num, inverse_propensity)
    mse = MSE_ips(preds, true, item, user_num, item_num, inverse_propensity)
    rmse = RMSE_ips(preds, true, item, user_num, item_num, inverse_propensity)

    return mae, mse, rmse


# propensity estimation for MF_IPS
def cal_propensity_score(user_num, item_num, ps_source_data_filename, ps_target_data_filename):
    ps_source_data = MF_DATA(ps_source_data_filename).data
    ps_source_data = ps_source_data.astype(int)
    ps_target_data = MF_DATA(ps_target_data_filename).data
    ps_target_data = ps_target_data.astype(int)

    P_L_TO = np.bincount(ps_source_data[:, 2], minlength=2)[:]
    tmp = P_L_TO.sum()
    P_L_TO = P_L_TO / P_L_TO.sum()

    P_L_T = np.bincount(ps_target_data[:, 2], minlength=2)[:]
    P_L_T = P_L_T / P_L_T.sum()

    P_O_T = tmp / (user_num * item_num)
    P = P_L_TO * P_O_T / P_L_T

    propensity_score = [P] * item_num

    return propensity_score