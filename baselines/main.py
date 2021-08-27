from config import opt
import os
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from metrics import AUC
from utils import MF_DATA, CausE_DATA, evaluate_model, cal_propensity_score
import numpy as np
import argparse
import random
import torch
import copy

seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# train for CausE
def train_CausE():
    train_data = CausE_DATA(opt.s_c_data, opt.s_t_data)
    val_data = MF_DATA(opt.cause_val_data)
    train_dataloader_s_c = DataLoader(train_data.s_c,
                                      opt.batch_size,
                                      shuffle=True)
    train_dataloader_s_t = DataLoader(train_data.s_t,
                                      opt.batch_size,
                                      shuffle=True)
    model = getattr(models,
                    opt.model)(train_data.users_num, train_data.items_num,
                               opt.embedding_size, opt.reg_c, opt.reg_c,
                               opt.reg_tc, train_data.s_c[:, :2].tolist(),
                               train_data.s_t[:, :2].tolist())

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in enumerate(train_dataloader_s_c):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % opt.verbose == 0:
            print('Epoch %d :' % (epoch))
            print('s_c Loss = ', loss.item())

        for i, data in enumerate(train_dataloader_s_t):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)

        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('s_t Loss = ', loss.item())
            print(
                'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                % (mae, mse, rmse, auc, time() - t1))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
          (best_iter, best_mae, best_mse, best_auc))

    best_model = getattr(models,
                         opt.model)(train_data.users_num, train_data.items_num,
                                    opt.embedding_size, opt.reg_c, opt.reg_c,
                                    opt.reg_tc, train_data.s_c[:, :2].tolist(),
                                    train_data.s_t[:, :2].tolist())
    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))

    print("\n========================= best model =========================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("===============================================================\n")

    return best_model


# train for MF_Naive and MF_IPS
def train():
    print('train begin')

    train_all_data = MF_DATA(opt.train_data)
    train_data = copy.deepcopy(train_all_data)
    val_data = MF_DATA(opt.val_all_data)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    if opt.model == 'MF_IPS':
        propensity_score = cal_propensity_score(train_data.users_num, train_data.items_num, opt.ps_source_data, opt.ps_target_data)
        propensity_score = np.array(propensity_score).astype(float)
        inverse_propensity = np.reciprocal(propensity_score)
        model = getattr(models, opt.model)(train_all_data.users_num,
                                           train_all_data.items_num,
                                           opt.embedding_size,
                                           inverse_propensity, opt.device)
    elif opt.model == 'MF_Naive':
        model = getattr(models, opt.model)(train_all_data.users_num,
                                           train_all_data.items_num,
                                           opt.embedding_size, opt.device)

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in enumerate(train_dataloader):
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(),
                                        label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()

        (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)

        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('Epoch %d [%.1f s]:'%(epoch, t2 - t1))
            print('Train Loss = ', loss.item())
            print(
                'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                % (mae, mse, rmse, auc, time() - t2))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
          (best_iter, best_mae, best_mse, best_auc))

    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(propensity_score)
        best_model = getattr(models, opt.model)(train_all_data.users_num,
                                                train_all_data.items_num,
                                                opt.embedding_size,
                                                inverse_propensity, opt.device)
    elif opt.model == 'MF_Naive':
        best_model = getattr(models, opt.model)(train_all_data.users_num,
                                                train_all_data.items_num,
                                                opt.embedding_size, opt.device)

    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))

    print("\n========================= best model =========================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("==============================================================\n")

    return best_model


def output_res(mae_output, mse_output, auc_output, mae, mse, auc):
    mae_output = np.append(mae_output, mae.detach().cpu().numpy())
    mse_output = np.append(mse_output, mse.detach().cpu().numpy())
    auc_output = np.append(auc_output, auc)
    mae_output = mae_output.reshape(-1, 1)
    mse_output = mse_output.reshape(-1, 1)
    auc_output = auc_output.reshape(-1, 1)
    return mae_output, mse_output, auc_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model', default='MF_Naive')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embedding_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--reg_c', type=float, default=0.1)
    parser.add_argument('--reg_t', type=float, default=0.001)
    parser.add_argument('--reg_tc', type=float, default=0.1)
    parser.add_argument('--metric', default='auc', choices=["mae", "mse", "auc"])

    args = parser.parse_args()
    opt.model = args.model
    opt.batch_size = args.batch_size
    opt.max_epoch = args.epoch
    opt.embedding_size = args.embedding_size
    opt.lr = args.lr
    opt.weight_decay = args.weight_decay
    opt.reg_c = args.reg_c
    opt.reg_t = args.reg_t
    opt.reg_tc = args.reg_tc
    opt.metric = args.metric

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    # setup_seed(seed_num)

    mae_output = np.array([])
    mse_output = np.array([])
    auc_output = np.array([])


    for cnt in range(50):
        if opt.model == 'MF_IPS' or opt.model == 'MF_Naive':
            best_model = train()
        elif opt.model == 'CausE':
            best_model = train_CausE()

        print("--------------------------test_1--------------------------")
        test_data = MF_DATA(opt.test_1_data)
        mae, mse, rmse, auc = evaluate_model(best_model, test_data, opt)
        print('MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
            (mae, mse, rmse, auc))
        mae_output, mse_output, auc_output = output_res(mae_output, mse_output, auc_output, mae, mse, auc)

        print("--------------------------test_2--------------------------")
        test_data = MF_DATA(opt.test_2_data)
        mae, mse, rmse, auc = evaluate_model(best_model, test_data, opt)
        print('MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
            (mae, mse, rmse, auc))
        mae_output, mse_output, auc_output = output_res(mae_output, mse_output, auc_output, mae, mse, auc)

        print("--------------------------test_3--------------------------")
        test_data = MF_DATA(opt.test_3_data)
        mae, mse, rmse, auc = evaluate_model(best_model, test_data, opt)
        print('MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
            (mae, mse, rmse, auc))
        mae_output, mse_output, auc_output = output_res(mae_output, mse_output, auc_output, mae, mse, auc)
        
        print("-------------------------test_all-------------------------")
        test_data = MF_DATA(opt.test_all_data)
        mae, mse, rmse, auc = evaluate_model(best_model, test_data, opt)
        print('MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
            (mae, mse, rmse, auc))
        mae_output, mse_output, auc_output = output_res(mae_output, mse_output, auc_output, mae, mse, auc)

    np.savetxt("./result/" + opt.model + "_res_50.txt", np.hstack((mae_output, mse_output, auc_output)))

    print('end')