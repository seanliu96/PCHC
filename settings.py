import os
import numpy as np


INF = np.finfo(np.float32).min / 2
EPS = 1e-12

log_dir = 'logs'

input_dir_20ng = 'datasets/20newsgroups'
input_dir_rcv1 = 'datasets/rcv1org'
input_dir_clef = 'datasets/CLEF'
input_dir_lshtc_small = 'datasets/LSHTC_small'
input_dir_lshtc_large = 'datasets/LSHTC_large2010'
input_dir_ipc = 'datasets/IPC'
input_dirs = [input_dir_20ng, input_dir_rcv1]

data_dir_20ng = 'data/20ng'
data_dir_rcv1 = 'data/rcv1'
data_dir_clef = 'data/clef'
data_dir_lshtc_small = 'data/lshtc_small'
data_dir_lshtc_large = 'data/lshtc_large'
data_dir_ipc = 'data/ipc'
data_dirs = [data_dir_20ng, data_dir_rcv1]

max_vocab_size = 100000
min_freq = 3
n_gram = 1
train_ratio = 0.8
label_ratios = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
times = 5


label_unlabel_idx_file = 'label_unlabel_idx.npz'
train_test_idx_file = 'train_test_idx.npz'
deltas_file = 'deltas.npy'
classes_file = 'classes.json'
vocab_file = 'vocab.json'
cat_hier_file = 'cat_hier.txt'
labeled_data_manager_file = 'labeled_data_manager.pkl'
unlabeled_data_manager_file = 'unlabeled_data_manager.pkl'
test_data_manager_file = 'test_data_manager.pkl'
labeled_svmlight_file = 'labeled_svmlight.txt'
dataless_svmlight_file = 'dataless_svmlight.txt'
test_svmlight_file = 'test_svmlight.txt'

# 0: macro
# 1: micro
main_metric=1

reduce_features = True

em_max_iter = 50
em_eps = 1e-3
path_weight = 1.0
pso_min_x = 0
pso_max_x = 2.0
pso_max_iter = 50
pso_group_size = 10
pso_patience = 5
'''
Poli, R., Kennedy, J., & Blackwell, T. (2007). Particle swarm optimization. Swarm intelligence, 1(1), 33-57.
https://pdfs.semanticscholar.org/4d2e/a871e1089efaea8feb0ce6c3b123304c5235.pdf
'''
pso_w = 0.7298
pso_c1 = 1.49618
pso_c2 = 1.49618

soft_sim=False