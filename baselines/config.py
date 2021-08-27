# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'
    is_eval_ips = False

    data_dir = './data'

    train_data = data_dir + '/train/extract_alldata.csv'
    val_all_data = data_dir + '/valid/validation.csv'
    test_1_data = data_dir + '/test/test_1.csv'
    test_2_data = data_dir + '/test/test_2.csv'
    test_3_data = data_dir + '/test/test_3.csv'
    test_all_data = data_dir + '/test/test.csv'

    # IPS data
    ps_source_data = data_dir + '/train/extract_bigtag.csv'
    ps_target_data = data_dir + '/train/extract_choicetag.csv'

    # CausE data
    s_c_data = data_dir + '/train/extract_bigtag.csv'
    s_t_data = data_dir + '/train/extract_choicetag.csv'
    cause_val_data = data_dir + '/valid/validation.csv'

    reg_c = 0.1
    reg_t = 0.001
    reg_tc = 0.1

    metric = 'mse'
    verbose = 50

    device = 'cpu'

    batch_size = 2048
    embedding_size = 40
    lr = 0.001
    weight_decay = 0.1

    max_epoch = 100

opt = DefaultConfig()


'''
MF_Naive
    batch_size = 2048
    embedding_size = 40
    lr = 0.001
    weight_decay = 0.1

MF_IPS
    batch_size = 256
    embedding_size = 48
    lr = 0.001
    weight_decay = 1

CausE
    batch_size = 2048
    embedding_size = 28
    lr = 0.005
    weight_decay = 0.1

    reg_c = 0.1
    reg_t = 0.001
    reg_tc = 0.1
'''