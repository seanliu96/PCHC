import logging
import logging.config
import logconfig
import numpy as np
import settings
import time
import tools
import os
import csv
import math
from multiprocessing import Process, Pool
from build_data_managers import DataManager
from scipy.sparse import vstack
from util import *
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from pso import PSO

def run_leaf(data_managers, deltas, C=1.0, method='LR_labeled'):
    logger = logging.getLogger(__name__)
    model_name = "flat" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    model_list = []
    test_pres = []
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    max_depth = len(sims)
    if 'LR' in method:
        model = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
    elif 'SVM' in method:
        model = LinearSVC(C=C, multi_class='crammer_singer')
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        tf_idf.fit(data_managers[0].xit)
        model.fit(tf_idf.transform(data_managers[0].xit), np.argmax(sims[-1], axis=1))
    else:
        model.fit(data_managers[0].xit, np.argmax(sims[-1], axis=1))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    if 'tf-idf' in method:
        test_pre = model.predict(tf_idf.transform(data_managers[2].xit))
    else:
        test_pre = model.predict(data_managers[2].xit)
    logger.info("predicting time: " + str(time.time() - start))
    return model, test_pre

def train_level(data_xit, sims, C=1.0, method='LR_labeled'):
    model_list = []
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        tf_idf.fit(data_xit)
    else:
        tf_idf = None
    for depth in range(len(sims)):
        if 'LR' in method:
            model = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
        elif 'SVM' in method:
            model = LinearSVC(C=C, multi_class='crammer_singer')
        if 'tf-idf' in method:
            model.fit(tf_idf.transform(data_xit), np.argmax(sims[depth], axis=1))
        else:
            model.fit(data_xit, np.argmax(sims[depth], axis=1))
        model_list.append(model)
    return model_list

def run_level(data_managers, deltas, C=1.0, method='LR_labeled'):
    logger = logging.getLogger(__name__)
    model_name = "level" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    model_list = []
    test_pres = []
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    model_list = train_level(data_managers[0].xit, sims, C, method)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    for depth in range(len(sims)):
        if 'tf-idf' in method:
            test_pre = model_list[depth].predict(tf_idf.transform(data_managers[2].xit))
        else:
            test_pre = model_list[depth].predict(data_managers[2].xit)
        test_pres.append(test_pre)
    logger.info("predicting time: " + str(time.time() - start))
    return model_list, test_pres

def train_one_depth(data_managers_list, depth, deltas, C=1.0, method='LR_labeled'):
    model_list = []
    # unlabeled_cnt = 0
    test_cnt = 0
    for i, data_managers in enumerate(data_managers_list):
        # unlabeled_cnt += data_managers[1].xit.shape[0] if data_managers[1] else 0
        test_cnt += data_managers[2].xit.shape[0]
    # unlabeled_pre = np.zeros((unlabeled_cnt,), dtype=np.int32)
    test_label = np.zeros((test_cnt,), dtype=np.int32)
    test_pre = np.zeros((test_cnt,), dtype=np.int32)
    for i, data_managers in enumerate(data_managers_list):
        if depth == 0:
            next_labels = np.array(range(deltas[0].shape[0]))
        else:
            next_labels = np.nonzero(deltas[depth - 1][i])[0]
        if 'labeled' in method:
            sim = data_managers[0].deltas[depth][:, next_labels]
        elif 'dataless' in method:
            if settings.soft_sim:
                if depth == 0:
                    sim = normalize(data_managers[0].sims[depth], axis=1)
                else:
                    sim = normalize(data_managers[0].sims[depth][:, next_labels], axis=1)
            else:
                if depth == 0:
                    sim = hardmax(data_managers[0].sims[depth], axis=1)
                else:
                    sim = hardmax(data_managers[0].sims[depth][:, next_labels], axis=1)
        else:
            raise NotImplementedError
        if 'LR' in method:
            model = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
        elif 'SVM' in method:
            model = LinearSVC(C=C, multi_class='crammer_singer')
        if 'tf-idf' in method:
            tf_idf = TfidfTransformer()
            tf_idf.fit(data_managers[0].xit)
            label = np.argmax(sim, axis=1)
            unique_label = np.unique(label)
            if len(unique_label) == 0:
                model = None
                # unlabeled_pre_part = np.zeros((data_managers[1].xit.shape[0], ), dtype=np.int32)
                test_pre_part = np.zeros((data_managers[2].xit.shape[0], ), dtype=np.int32)
            elif len(unique_label) == 1:
                model = None
                # unlabeled_pre_part = np.array([unique_label[0]] * (data_managers[1].xit.shape[0] if data_managers[1] else 0))
                test_pre_part = np.array([unique_label[0]] * data_managers[2].xit.shape[0])
            else:
                model.fit(tf_idf.transform(data_managers[0].xit), label)
                # if data_managers[1].xit.shape[0] > 0:
                #     unlabeled_pre_part = model.predict(tf_idf.transform(data_managers[1].xit))
                # else:
                #     unlabeled_pre_part = []
                if data_managers[2].xit.shape[0] > 0:
                    test_pre_part = model.predict(tf_idf.transform(data_managers[2].xit))
                else:
                    test_pre_part = []
        else:
            tf_idf = None
            label = np.argmax(sim, axis=1)
            unique_label = np.unique(label)
            if len(unique_label) == 0:
                model = None
                # unlabeled_pre_part = np.zeros((data_managers[1].xit.shape[0], ), dtype=np.int32)
                test_pre_part = np.zeros((data_managers[2].xit.shape[0], ), dtype=np.int32)
            elif len(unique_label) == 1:
                model = None
                # unlabeled_pre_part = np.array([unique_label[0]] * (data_managers[1].xit.shape[0] if data_managers[1] else 0))
                test_pre_part = np.array([unique_label[0]] * data_managers[2].xit.shape[0])
            else:
                model.fit(data_managers[0].xit, label)
                # if data_managers[1].xit.shape[0] > 0:
                #     unlabeled_pre_part = model.predict(data_managers[1].xit)
                # else:
                #     unlabeled_pre_part = []
                if data_managers[2].xit.shape[0] > 0:
                    test_pre_part = model.predict(data_managers[2].xit)
                else:
                    test_pre_part = []
        model_list.append(model)
        if depth != 0:
            # unlabeled_pre_part = np.array([next_labels[x] for x in unlabeled_pre_part])
            test_pre_part = np.array([next_labels[x] for x in test_pre_part])
        # if data_managers[1].true_idx is None:
        #     unlabeled_pre = unlabeled_pre_part
        # elif len(data_managers[1].true_idx) > 0:
        #     unlabeled_pre[data_managers[1].true_idx] = unlabeled_pre_part
        if data_managers[2].true_idx is None:
            test_label = data_managers[2].labels[depth]
            test_pre = test_pre_part
        elif len(data_managers[2].true_idx) > 0:
            test_label[data_managers[2].true_idx] = data_managers[2].labels[depth]
            test_pre[data_managers[2].true_idx] = test_pre_part
    return model_list, None, test_pre

def run_TD(data_managers, deltas, C=1.0, method='LR_labeled'):
    logger = logging.getLogger(__name__)
    if 'BU' in method:
        model_name = method
    else:
        model_name = 'TD' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    model_lists = []
    unlabeled_pres = []
    test_pres = []
    if 'labeled' in method:
        labels = data_managers[0].labels
    elif 'dataless' in method:
        labels = list(map(lambda sim: np.argmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    max_depth = len(deltas)
    
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        tf_idf.fit(data_managers[0].xit)
    else:
        tf_idf = None
    data_managers_d0 = [
        DataManager(data_managers[0].name + '_d0', xit=data_managers[0].xit, labels=data_managers[0].labels,
                    deltas=data_managers[0].deltas, sims=data_managers[0].sims, true_idx=None),
        None,
        DataManager(data_managers[2].name + '_d0', xit=data_managers[2].xit, labels=data_managers[2].labels,
                    deltas=data_managers[2].deltas, sims=data_managers[2].sims, true_idx=None)]
    data_managers_list = [data_managers_d0]
    for depth in range(max_depth):
        model_list, unlabeled_pre, test_pre = train_one_depth(data_managers_list, depth, deltas, C, method)
        model_lists.append(model_list)
        # unlabeled_pres.append(unlabeled_pre)
        test_pres.append(test_pre)
        # prepare for the next depth
        if depth == max_depth - 1:
            break
        class_depth_no = deltas[depth].shape[0]
        labeled_true_idx_list = [[] for i in range(class_depth_no)]
        # unlabeled_true_idx_list = [[] for i in range(class_depth_no)]
        test_true_idx_list = [[] for i in range(class_depth_no)]

        for i, l in enumerate(labels[depth]):
            labeled_true_idx_list[l].append(i)
        # for i, u in enumerate(unlabeled_pre):
        #     unlabeled_true_idx_list[u].append(i)
        for i, t in enumerate(test_pre):
            test_true_idx_list[t].append(i)
        data_managers_list.clear()
        for i in range(class_depth_no):
            data_managers_list.append([
                build_subdata_manager(data_managers_d0[0], data_managers[0].name + '_d%d_c%d' % (depth, i),
                                      labeled_true_idx_list[i]),
                None,
                build_subdata_manager(data_managers_d0[2], data_managers[2].name + '_d%d_c%d' % (depth, i),
                                      test_true_idx_list[i])])
    logger.info("training and predicting time: " + str(time.time() - start))
    return model_lists, test_pres

def run_BU(data_managers, deltas, C=1.0, method='LR_labeled'):
    reverse_data_managers = [
        build_reverse_data_manager(data_managers[0], data_managers[0].name + '_reversed'),
        None,
        build_reverse_data_manager(data_managers[2], data_managers[2].name + '_reversed')]
    reverse_deltas = list(map(lambda delta: delta.T, deltas[0:-1]))
    reverse_deltas.reverse()
    reverse_deltas.append(np.zeros((0, 0), dtype=np.int32))
    model_lists, test_pres = run_TD(
        reverse_data_managers, reverse_deltas, C, method='BU' + method)
    return model_lists[::-1], test_pres[::-1]

def train_WD(data_xit, sims, C=1.0, method='LR_labeled'):
    model_list = []
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        tf_idf.fit(data_xit)
    else:
        tf_idf = None
    for depth in range(len(sims)):
        if 'LR' in method:
            model = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
        elif 'SVM' in method:
            model = CalibratedClassifierCV(LinearSVC(C=C, multi_class='crammer_singer'))
        if 'tf-idf' in method:
            model.fit(tf_idf.transform(data_xit), np.argmax(sims[depth], axis=1))
        else:
            model.fit(data_xit, np.argmax(sims[depth], axis=1))
        model_list.append(model)
    return model_list

def predict_prob_WD_pathscore(model_list, data_xit, deltas=None, path_weights=None):
    P_xcs = []
    classes_size = []
    N = data_xit.shape[0]
    for depth in range(len(model_list)):
        model = model_list[depth]
        P_xc = model.predict_proba(data_xit)
        P_xcs.append(P_xc * path_weights[depth])
        classes_size.append(P_xc.shape[1])
    if N > 0:
        P_xcs_matrix = P_xcs[0]
        for depth in range(1, len(P_xcs)):
            P_xcs_matrix = np.add(np.expand_dims(P_xcs_matrix, axis=2), np.expand_dims(P_xcs[depth], axis=1))
            P_xcs_matrix = P_xcs_matrix.reshape((N, -1))
        if deltas is not None:
            delta_matrix = deltas[0]
            for depth in range(1, len(deltas) - 1):
                delta_matrix = np.multiply(np.expand_dims(delta_matrix, axis=2), np.expand_dims(deltas[depth], axis=0))
                delta_matrix = delta_matrix.reshape((-1, deltas[depth].shape[-1]))
            delta_matrix = delta_matrix.reshape((1, -1))
            try:
                P_xcs_matrix = P_xcs_matrix * delta_matrix
            except:
                pass
        P_xcs_matrix = normalize(P_xcs_matrix, axis=1)
        P_xcs_matrix = P_xcs_matrix.reshape([N] + classes_size)  # N * C1 * C2 ... * Cn
    else:
        P_xcs_matrix = np.zeros([0] + classes_size)
    return P_xcs_matrix

def predict_WD_pathscore(model_list, data_xit, deltas=None, C=1.0, path_weights=None):
    P_xcs_matrix = predict_prob_WD_pathscore(model_list, data_xit, deltas=deltas, path_weights=path_weights)
    N = P_xcs_matrix.shape[0]
    classes_size = P_xcs_matrix.shape[1:]
    Ps = []
    for class_size in classes_size:
        Ps.append(np.zeros((N, class_size)))
    for i in range(N):
        max_index_list = np.unravel_index(np.argmax(P_xcs_matrix[i]), classes_size)
        for j in range(len(Ps)):
            Ps[j][i, max_index_list[j]] = 1
    return Ps

def predict_label_WD_pathscore(model_list, data_xit, deltas=None, C=1.0, path_weights=None):
    data_xit_csr = data_xit.tocsr()
    y_pres = [None for k in range(len(model_list))]
    batch_size = 512
    for i in range(0, data_xit.shape[0], batch_size):
        j = min(i + batch_size, data_xit.shape[0])
        Ps = predict_WD_pathscore(model_list, data_xit_csr[i:j, :], deltas=deltas, path_weights=path_weights)
        for k in range(len(Ps)):
            if y_pres[k] is None:
                y_pres[k] = np.argmax(Ps[k], axis=1)
            else:
                y_pres[k] = np.concatenate([y_pres[k], np.argmax(Ps[k], axis=1)])
    return y_pres

def run_WD(data_managers, deltas, C=1.0, method='LR_labeled', soft_pathscore=True, path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = 'WD_' + ("soft_" if soft_pathscore else "hard_") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    model_list = train_WD(data_managers[0].xit, sims, C, method)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    test_pres = predict_label_WD_pathscore(model_list, data_managers[2].xit,
                                            deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
    logger.info("predicting time: " + str(time.time() - start))
    return model_list, test_pres

def run_PSO_WD(data_managers, deltas, C=1.0, method='LR_labeled', soft_pathscore=True, path_weights=None, nos=None):
    logger = logging.getLogger(__name__)
    model_name = 'WD(PSO)_' + ("soft_" if soft_pathscore else "hard_") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
        labels = data_managers[0].labels
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
        labels = list(map(lambda sim: np.argmax(sim, axis=-1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    model_list = train_WD(data_managers[0].xit, sims, C, method)
    def score_function(path_weights):
        labeled_pres = predict_label_WD_pathscore(model_list, data_managers[0].xit,
                                            deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
        return compute_overall_p_r_f1(labels, labeled_pres, nos)[2][settings.main_metric]
    pso = PSO(path_weights, score_function, group_size=settings.pso_group_size, min_x=settings.pso_min_x, max_x=settings.pso_max_x)
    pso.update(c1=settings.pso_c1, c2=settings.pso_c2, w=settings.pso_w, max_iter=settings.pso_max_iter, patience=settings.pso_patience)
    path_weights = pso.get_best_x()
    logger.info("training time: " + str(time.time() - start))
    logger.info('best_path_weight: %s' % (str(path_weights)))
    start = time.time()
    test_pres = predict_label_WD_pathscore(model_list, data_managers[2].xit,
                                            deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
    logger.info("predicting time: " + str(time.time() - start))
    return model_list, test_pres

def train_PC(data_xit, path_score, C=1.0, method='LR_labeled'):
    
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        tf_idf.fit(data_managers[0].xit)
    else:
        tf_idf = None
    if 'LR' in method:
        model = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial')
    elif 'SVM' in method:
        model = CalibratedClassifierCV(LinearSVC(C=C, multi_class='crammer_singer'))
    data_xit_csr = data_xit.tocsr()
    data_xit_expanded = []
    sample_weight = []
    label_expanded = []
    for i in range(path_score.shape[0]):
        for j in range(path_score.shape[1]):
            if path_score[i][j] > 0:
                data_xit_expanded.append(data_xit_csr.getrow(i))
                sample_weight.append(path_score[i][j])
                label_expanded.append(j)
    data_xit_expanded = vstack(data_xit_expanded)
    if 'tf-idf' in method:
        model.fit(tf_idf.transform(data_xit_expanded), label_expanded, sample_weight=sample_weight)
    else:
        model.fit(data_xit_expanded, label_expanded, sample_weight=sample_weight)
    return model

def predict_PC_pathscore(model, data_xit, deltas):
    Ps = []
    P_bottom = model.predict_proba(data_xit)
    Ps.append(P_bottom)
    for depth in range(len(deltas) - 2, -1, -1):
        try:
            P_high = normalize(np.dot(P_bottom, deltas[depth].T), axis=1)
        except ValueError:
            P_high = normalize(np.ones((P_bottom.shape[0], deltas[depth].shape[0])), axis=1)
        Ps.append(P_high)
        P_bottom = P_high
    Ps.reverse()
    return Ps

def predict_label_PC_pathscore(model, data_xit, deltas):
    Ps = predict_PC_pathscore(model, data_xit, deltas)
    y_pres = []
    y_bottom = np.argmax(Ps[-1], axis=1)
    y_pres.append(y_bottom)
    for depth in range(len(deltas) - 2, -1, -1):
        y_high = [np.argmax(deltas[depth][:, j]) for j in y_bottom]
        y_bottom = y_high
        y_pres.append(y_bottom)
    y_pres.reverse()
    return y_pres

def run_PC(data_managers, deltas, C=1.0, method='LR_labeled', path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = "PC" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    path_score = compute_path_score(sims, deltas, path_weights=path_weights)
    model = train_PC(data_managers[0].xit, path_score, C, method)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    test_pres = predict_label_PC_pathscore(model, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return model, test_pres

def run_PSO_PC(data_managers, deltas, C=1.0, method='LR_labeled', path_weights=None, nos=None):
    logger = logging.getLogger(__name__)
    model_name = "PC(PSO)" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
        labels = data_managers[0].labels
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
        labels = list(map(lambda sim: np.argmax(sim, axis=-1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    def score_function(path_weights):
        path_score = compute_path_score(sims, deltas, path_weights=path_weights)
        model = train_PC(data_managers[0].xit, path_score, C, method)
        labeled_pres = predict_label_PC_pathscore(model, data_managers[0].xit, deltas)
        return compute_overall_p_r_f1(labels, labeled_pres, nos)[2][settings.main_metric]
    pso = PSO(path_weights, score_function, group_size=settings.pso_group_size, min_x=settings.pso_min_x, max_x=settings.pso_max_x)
    pso.update(c1=settings.pso_c1, c2=settings.pso_c2, w=settings.pso_w, max_iter=settings.pso_max_iter, patience=settings.pso_patience)
    path_weights = pso.get_best_x()
    path_score = compute_path_score(sims, deltas, path_weights=path_weights)
    model = train_PC(data_managers[0].xit, path_score, C, method)
    logger.info("training time: " + str(time.time() - start))
    logger.info('best_path_weight: %s' % (str(path_weights)))
    start = time.time()
    test_pres = predict_label_PC_pathscore(model, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return model, test_pres

def run_classifiers(classifier_name, data_managers, method, **kw):    
    if 'labeled' in method:
        labels = check_labels(data_managers[0].labels[-1])
        if len(labels) == 1:
            return None, [labels[0]] * data_managers[2].xit.shape[0]
    elif 'dataless' in method and 'leaf' in classifier_name:
        labels = check_labels(np.argmax(data_managers[0].sims[-1], axis=1))
        if len(labels) == 1:
            return None, [labels[0]] * data_managers[2].xit.shape[0]
    
    if 'LR' in classifier_name:
        if classifier_name == 'flatLR':
            return run_leaf(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method)
        elif classifier_name == 'levelLR':
            return run_level(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method)
        elif classifier_name == 'TDLR':
            return run_TD(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method)
        elif classifier_name == 'BULR':
            return run_BU(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method)
        elif classifier_name == 'WDLR_soft':
            return run_WD(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, soft_pathscore=True, path_weights=kw['path_weights'])
        elif classifier_name == 'WDLR(PSO)_soft':
            return run_PSO_WD(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, soft_pathscore=True, path_weights=kw['path_weights'], nos=kw['nos'])
        elif classifier_name == 'WDLR_hard':
            return run_WD(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, soft_pathscore=False, path_weights=kw['path_weights'])
        elif classifier_name == 'WDLR(PSO)_hard':
            return run_PSO_WD(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, soft_pathscore=False, path_weights=kw['path_weights'], nos=kw['nos'])
        elif classifier_name == 'PCLR':
            return run_PC(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, path_weights=kw['path_weights'])
        elif classifier_name == 'PCLR(PSO)':
            return run_PSO_PC(data_managers, kw['deltas'], C=kw['C'], method='LR_'+method, path_weights=kw['path_weights'], nos=kw['nos'])
    elif 'SVM' in classifier_name:
        if classifier_name == 'flatSVM':
            return run_leaf(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method)
        elif classifier_name == 'levelSVM':
            return run_level(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method)
        elif classifier_name == 'TDSVM':
            return run_TD(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method)
        elif classifier_name == 'BUSVM':
            return run_BU(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method)
        elif classifier_name == 'WDSVM_soft':
            return run_WD(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, soft_pathscore=True, path_weights=kw['path_weights'])
        elif classifier_name == 'WDSVM(PSO)_soft':
            return run_PSO_WD(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, soft_pathscore=True, path_weights=kw['path_weights'], nos=kw['nos'])
        elif classifier_name == 'WDSVM_hard':
            return run_WD(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, soft_pathscore=False, path_weights=kw['path_weights'])
        elif classifier_name == 'WDSVM(PSO)_hard':
            return run_PSO_WD(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, soft_pathscore=False, path_weights=kw['path_weights'], nos=kw['nos'])
        elif classifier_name == 'PCSVM':
            return run_PC(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, path_weights=kw['path_weights'])
        elif classifier_name == 'PCSVM(PSO)':
            return run_PSO_PC(data_managers, kw['deltas'], C=kw['C'], method='SVM_'+method, path_weights=kw['path_weights'], nos=kw['nos'])
    raise NotImplementedError('%s is not supported!' % (classifier_name))
        
def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, times=1, classifier_names=None, C=1.0):
    logger = logging.getLogger(__name__)

    if label_ratio == 1.0:
        times = 1
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    deltas = tools.load(os.path.join(input_dir, settings.deltas_file))
    
    if not classifier_names:
        classifier_names = ['flatLR', 'levelLR',
                            'flatSVM', 'levelSVM']
    path_weights = [1.0]
    for i in range(1, len(classes)):
        path_weights.append(path_weights[-1] * settings.path_weight)
    path_weights = np.asarray(path_weights)
    nos, hier_tree = get_hier_info(input_dir)
    kw = {'deltas': deltas, 'path_weights': path_weights, 'nos': nos}
    if label_ratio == 1.0:
        times = 1
    for mode in ["labeled", "dataless"]:
        metrics_result = np.zeros((times, 2, len(classifier_names)*len(settings.Cs), 3, 2)) # times, methods, depth+1, [[(M_precision,m_precision), (M_recall,m_recall),  (M_f1, m_f1)], ...]

        for i in range(times):
            method_index = 0
            sub_dir = os.path.join(input_dir, str(label_ratio), str(i))
            logger.info(logconfig.key_log(logconfig.START_PROGRAM, sub_dir))

            data_managers = load_data_managers(sub_dir)
            if settings.reduce_features:
                non_zero_indices = np.nonzero(data_managers[0].xit)
                non_zero_columns = sorted(set(non_zero_indices[1]))
                for data_manager in data_managers:
                    data_manager.xit = data_manager.xit[:,non_zero_columns]

            if mode == "dataless" and np.max(data_managers[2].sims[0][0]) == 0.0:
                continue

            for j, classifier_name in enumerate(classifier_names):
                for k, C in enumerate(settings.Cs):
                    kw['C'] = C
                    result = run_classifiers(classifier_name, data_managers, mode, **kw)
                    if len(data_managers[2].labels) == len(result[1]):
                        metrics_result[i,0,j*len(settings.Cs)+k] = compute_p_r_f1(data_managers[2].labels[-1], result[1][-1])
                        metrics_result[i,1,j*len(settings.Cs)+k] = compute_overall_p_r_f1(data_managers[2].labels, result[1], nos)
                    else:
                        metrics_result[i,0,j*len(settings.Cs)+k] = compute_p_r_f1(data_managers[2].labels[-1], result[1])
                        metrics_result[i,1,j*len(settings.Cs)+k] = compute_hier_p_r_f1(data_managers[2].labels[-1], result[1], nos, hier_tree)

        avg_M_metrics_result = np.mean(metrics_result[:,:,:,:,0], axis=0)
        std_M_metrics_result = np.std(metrics_result[:,:,:,:,0], axis=0)
        avg_m_metrics_result = np.mean(metrics_result[:,:,:,:,1], axis=0)
        std_m_metrics_result = np.std(metrics_result[:,:,:,:,1], axis=0)
        
        headers = []
        for j, classifier_name in enumerate(classifier_names):
            for k, C in enumerate(settings.Cs):
                headers.append('%s_C_%.2f' % (classifier_name, C))
        with open(os.path.join(input_dir, str(label_ratio), 'LR_SVM_%s.csv' % (mode)), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Leaf'] + headers)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[0,:,0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[0,:,0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[0,:,0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[0,:,0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[0,:,1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[0,:,1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[0,:,1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[0,:,1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[0,:,2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[0,:,2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[0,:,2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[0,:,2]))
            csv_writer.writerow([])
            csv_writer.writerow(['Overall'] + headers)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[1,:,0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[1,:,0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[1,:,0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[1,:,0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[1,:,1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[1,:,1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[1,:,1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[1,:,1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[1,:,2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[1,:,2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[1,:,2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[1,:,2]))
            csv_writer.writerow([])
    logger.info(logconfig.key_log(logconfig.END_PROGRAM, sub_dir))
    
if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'LR_SVM.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))

    classifier_names = ['flatLR', 'TDLR', 'WDLR_hard', 'PCLR', 
                            'flatSVM', 'TDSVM']

    pool = Pool(20)
    for input_dir in settings.data_dirs:
        for label_ratio in settings.label_ratios:
            pool.apply_async(main, args=(input_dir, label_ratio, settings.times, classifier_names))
            # main(input_dir, label_ratio, settings.times, classifier_names)
    pool.close()
    pool.join()
