from tifffile import imread, imwrite
from skimage import filters
import subprocess
import pandas as pd
import os
from datetime import datetime
import numpy as np
import argparse
import sys
import time
from collections import defaultdict
import cc3d
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
import napari
import pickle
import warnings
import napari.utils.notifications as notif
from napari.qt import create_worker
from gui import *
from time import sleep

warnings.filterwarnings("ignore", category=FutureWarning)
from gudhi.clustering.tomato import Tomato
from napari.settings import get_settings

settings = get_settings()
settings.application.window_size = (2560, 1440)


# settings.application.window_maximized = False
# settings.application.window_fullscreen = False

# v4 split clusters with 2 nuclei

# parser = argparse.ArgumentParser()

# parser.add_argument('-b', '--f-data', dest='f_data', type=str, default='b')
# parser.add_argument('-f', dest='f_data', type=str, default=None)

# parser = argparse.ArgumentParser(description='Microglia Cell Segmentation')
#
# parser.add_argument('-f', '--f-data', dest='f_data', type=str, default=None)
# parser.add_argument('-sd', '--save-dir', dest='save_dir', type=str, default='predictions/')
# parser.add_argument('-ds', '--down-sample', dest='ds', nargs='+', type=int, default=None)
# parser.add_argument('-inv', '--invert', dest='inv', type=bool, default=False)
# parser.add_argument('-scale', '--scale', dest='scale', type=bool, default=False)
# parser.add_argument('-delta', '--delta', dest='delta', type=float, default=3)
#
# parser.add_argument('-oc', '--otsu-cell', dest='otsu_cell', type=float, default=0.5)
# parser.add_argument('-on', '--otsu-nuclei', dest='otsu_nuclei', type=float, default=0.5)
# parser.add_argument('-rc', '--radius-cell', dest='radius_cell', type=float, default=1.8)
# parser.add_argument('-rn', '--radius-nuclei', dest='radius_nuclei', type=float, default=1.8)
# parser.add_argument('-pc', '--persi-cell', dest='persi_cell', type=float, default=0.5)
# parser.add_argument('-pn', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)

# parser = argparse.ArgumentParser(description='Interactive Persistence-based Clustering')
# parser.add_argument('-a', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)
# parser.add_argument('-b', '--persi-nuclei', dest='persi_nuclei', type=float, default=0.3)

# parser.add_argument('-f', '--f-data', dest='f_data', type=str, default=None)

# parser = argparse.ArgumentParser()
# parser.add_argument('--f', type=str, default=None, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls',
#                     help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=1, help='Decay rate for lr decay [default: 0.8]')
# FLAGS = parser.parse_args()

# args = vars(parser.parse_args())
# print(args)

# if FLAGS.f_data is None:
#     print('give input path by \'-f path-to-tiff-file\'')
#     sys.exit()


# fname = FLAGS.f_data.split('/')[-1]
#
# FLAGS.otsu_cell = FLAGS.otsu_cell
# FLAGS.otsu_nuclei = FLAGS.otsu_nuclei
# FLAGS.radius_cell = str(FLAGS.radius_cell)
# FLAGS.radius_nuclei = str(FLAGS.radius_nuclei)
# FLAGS.persi_cell = str(FLAGS.persi_cell)
# FLAGS.persi_nuclei = str(FLAGS.persi_nuclei)
#
# print('otsu cell: %.1f' % FLAGS.otsu_cell)
# print('radius cell: %.1f' % FLAGS.radius_cell)
# print('persi cell: %.1f' % FLAGS.persi_cell)


# if not os.path.isdir(FLAGS.save_dir):
#     os.mkdir(FLAGS.save_dir)
#
# if FLAGS.f_data is None:
#     print('please input the input path by \'-f path-to-tiff-file\'')
#     sys.exit()


def down_sample_cc(x, ds):
    """

    :param x: z * c * x * y
    :param ds:
    :return:
    """
    x_ds = np.arange(0, x.shape[0], ds[0])
    y_ds = np.arange(0, x.shape[1], ds[1])
    z_ds = np.arange(0, x.shape[2], ds[2])
    x = x[x_ds, :, :]
    x = x[:, y_ds, :]
    x = x[:, :, z_ds]
    return x


def down_sample(x, ds):
    """

    :param x: z * c * x * y
    :param ds:
    :return:
    """
    x_ds = np.arange(0, x.shape[0], ds[0])
    y_ds = np.arange(0, x.shape[2], ds[1])
    z_ds = np.arange(0, x.shape[3], ds[2])
    x = x[x_ds, :, :, :]
    x = x[:, :, y_ds, :]
    x = x[:, :, :, z_ds]
    return x


def remove_by_num_points(x, num_points):
    values = np.sort(np.reshape(x, [-1]))
    th = values[-num_points + 1]
    x[np.where(x < th)] = 0
    return x


def otsu_thresholding_nuclei(x, adj, num_points):
    idx_mid_slice = int(x.shape[0] / 2)
    val = filters.threshold_otsu(x[idx_mid_slice]) * adj
    print('otsu: %.2f' % val)
    idx = np.where(x < val)
    if len(idx[0]) > num_points:
        values = np.sort(np.reshape(x, [-1]))
        th = values[-num_points + 1]
        x[np.where(x < th)] = 0
    else:
        x[idx] = 0
    return x


def otsu_thresholding(x, adj):
    x_otsu = np.zeros_like(x)
    for i in range(len(x)):
        plane = x[i]
        if np.max(plane) != np.min(plane):
            val = filters.threshold_otsu(plane) * adj
            plane[np.where(plane < val)] = 0
        x_otsu[i] = plane
    return x_otsu

    # idx_mid_slice = int(x.shape[0] / 2)
    # val = filters.threshold_otsu(x[idx_mid_slice]) * adj
    # print('otsu: %.2f' % val)
    # idx = np.where(x < val)
    # x[idx] = 0
    # return x


def otsu_value(x, adj):
    idx_mid_slice = int(x.shape[0] / 2)
    val = filters.threshold_otsu(x[idx_mid_slice]) * adj
    return val


def pad_or_trim(x, n):
    if len(x) < n:
        x_ = np.zeros([n, x.shape[-1]])
        x_[:len(x)] = x
    elif len(x) > n:
        perm = np.random.permutation(len(x))[:n]
        x_ = x[perm]
    else:
        x_ = x
    return x_


def radius_unit(scale_dims):
    return str(np.sqrt(np.sum(np.array(scale_dims) ** 2)))


def sort_pred_by_radius_2d(pred):
    s = []
    cc_idx = np.unique(pred).tolist()[1:]
    for i in cc_idx:
        idx = np.where(pred == i)
        dx = np.max(idx[0]) - np.min(idx[0])
        dy = np.max(idx[1]) - np.min(idx[1])
        s.append(max(dx, dy))
    # return sorted(cc_idx, key=lambda k: s[k], reverse=True)
    idx_sorted = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    return np.array(cc_idx)[idx_sorted]


def sort_pred_by_radius(pred):
    s = []
    cc_idx = np.unique(pred).tolist()[1:]
    for i in cc_idx:
        idx = np.where(pred == i)
        dx = np.max(idx[1]) - np.min(idx[1])
        dy = np.max(idx[2]) - np.min(idx[2])
        s.append(max(dx, dy))
    # return sorted(cc_idx, key=lambda k: s[k], reverse=True)
    idx_sorted = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    return np.array(cc_idx)[idx_sorted]


# def sort_pred_by_radius_cc(pred):
#     s = []
#     cc_idx = np.unique(pred).tolist()[1:]
#     for i in cc_idx:
#         idx = np.where(pred == i)
#         dx = np.max(idx[1]) - np.min(idx[1])
#         dy = np.max(idx[2]) - np.min(idx[2])
#         s.append(max(dx, dy))
#     return sorted(range(len(s)), key=lambda k: s[k], reverse=True)


def remove_small_clusters_2d(pred, max_num_clusters):
    # ds = [2, 4, 4]
    # pred = down_sample_cc(pred, ds)
    # print('%d clusters' % np.max(pred))
    # sorted_pred_class = np.array(sort_pred(pred)) + 1  # sort the cluster index from large to small
    sorted_pred_class = np.array(sort_pred_by_radius_2d(pred))  # sort the cluster index from large to small
    pred_removed = np.zeros(pred.shape)
    for i in range(max_num_clusters):
        pred_removed[np.where(pred == sorted_pred_class[i])] = i + 1
    return pred_removed


def pick_top_k_clusters(preds, k):
    print(k)
    # print('creating 3d nuclei masks')
    ds = [2, 4, 4]
    preds_ds = down_sample_cc(preds, ds)
    cc_idx = np.unique(preds_ds).tolist()[1:]
    # print('%d components' % np.max(cc_idx))
    dic = {}
    for idx in cc_idx:
        # print('removing by size %d/%d' % (idx, np.max(cc_idx)))
        size = len(np.where(preds_ds == idx)[0])
        dic[idx] = size
    sorted_list = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_list[:10])
    pred_removed = np.zeros(preds.shape)

    for i in range(k):
        pred_removed[np.where(preds == sorted_list[i][0])] = i + 1

    return pred_removed


def remove_by_size(preds, th):
    ds = [2, 4, 4]
    preds_ds = down_sample_cc(preds, ds)
    cc_idx = np.unique(preds_ds).tolist()[1:]
    idx_list = []
    for idx in cc_idx:
        # print('removing by size %d/%d' % (idx, np.max(cc_idx)))
        size = len(np.where(preds_ds == idx)[0])
        if size > th / np.prod(ds):
            idx_list.append(idx)
    pred_removed = np.zeros(preds.shape)
    for i in range(len(idx_list)):
        pred_removed[np.where(preds == idx_list[i])] = i + 1
    return pred_removed


def remove_small_clusters(pred, max_num_clusters):
    ds = [2, 4, 4]
    pred = down_sample_cc(pred, ds)
    # print('%d clusters' % np.max(pred))
    # sorted_pred_class = np.array(sort_pred(pred)) + 1  # sort the cluster index from large to small
    sorted_pred_class = np.array(sort_pred_by_radius(pred))  # sort the cluster index from large to small
    pred_removed = np.zeros(pred.shape)
    for i in range(max_num_clusters):
        pred_removed[np.where(pred == sorted_pred_class[i])] = i + 1
    return pred_removed


# def remove_small_clusters_cc(pred, max_num_clusters):
#     ds = [2, 4, 4]
#     pred = down_sample_cc(pred, ds)
#     # print('%d clusters' % np.max(pred))
#     # sorted_pred_class = np.array(sort_pred(pred)) + 1  # sort the cluster index from large to small
#     sorted_pred_class = np.array(sort_pred_by_radius(pred)) + 1 # sort the cluster index from large to small
#     pred_removed = np.zeros(pred.shape)
#     for i in range(max_num_clusters):
#         pred_removed[np.where(pred == sorted_pred_class[i])] = i + 1
#     return pred_removed


def visualize_pbc_2d(pred):
    color_dic = mcolors.CSS4_COLORS
    keys_all = [k for k in color_dic.keys() if k != 'white' and k != 'snow']
    num_labels = int(np.max(pred))
    perm = np.random.permutation(len(keys_all))[:num_labels]
    keys = [keys_all[i] for i in perm]

    mask = np.zeros(list(pred.shape) + [3])
    for i in range(1, num_labels + 1):
        vec = mcolors.to_rgb(color_dic[keys[i - 1]])
        mask[np.where(pred == i)] = vec
    mask[np.where(mask == 0)] = 255
    img = Image.fromarray(mask)
    img.save('nuclei.png')


def preprocessing(x, invert, otsu_coef, scale):
    if np.min(x) != 0:
        x = x - np.min(x)
    if invert:
        x = np.max(x) - x

    x = otsu_thresholding(x, adj=otsu_coef)

    if scale:
        x = x / np.max(x)
    return x


def preprocessing_v2(x, invert, otsu_coef, scale):
    if np.min(x) != 0:
        x = x - np.min(x)
    if invert:
        x = np.max(x) - x

    v = otsu_value(x, adj=otsu_coef)

    x[np.where(x < v)] = 0

    if scale:
        x = x / np.max(x)

    return x, v


def pbc_2d(x, f_pointcloud, radius, th, max_num_classes=None):
    """

    :param data_path: path to the tiff file
    :return: point_set_path: path to the point set defined by the user
    """

    # x = imread(f_data)

    # f_data_otsu = f_data.split('.')[0] + '-otsu.' + f_data.split('.')[1]
    # imwrite(f_data_otsu, x)

    idx = np.where(x != 0)
    with open(f_pointcloud, 'w') as f:
        for i in range(len(idx[0])):
            line = str(idx[0][i]) + " " + str(idx[1][i]) + " " + str(
                x[idx[0][i], idx[1][i]]) + "\n"
            f.write(line)

    print('pointcloud file generated! %d points' % len(idx[0]))
    subprocess.run(["./tomatoyue", f_pointcloud, radius, th])
    pred_tiff = np.zeros(x.shape)
    pred = []
    with open('clusters.txt') as f:
        for line in f:
            if line[:-1] == 'NaN':
                pred.append(0)
            else:
                pred.append(int(line[:-1]))
    pred = np.array(pred)
    x = np.fromfile(f_pointcloud, dtype=np.float,
                    sep=" ")
    x = np.reshape(x, (-1, 3))
    for i in range(len(x)):
        pred_tiff[int(x[i, 0]), int(x[i, 1])] = pred[i]

    # np.save(f_pred.split('.tif')[0], pred_tiff)
    # print('prediction saved as numpy array')
    num_classes = np.max(pred_tiff)
    #
    if max_num_classes is not None and num_classes > max_num_classes:
        print('removing small clusters...')
        pred_tiff = remove_small_clusters(pred_tiff, max_num_clusters=max_num_classes)
    return pred_tiff


def pbc_mid_tau(x, idx, idx_coef=0.75):
    """

    :param data_path: path to the tiff file
    :return: point_set_path: path to the point set defined by the user
    """
    density = x[idx]
    t = Tomato(graph_type='radius',
               r=1.8,
               metric='euclidean',
               density_type='manual',
               merge_threshold=np.max(density) + 1,
               )
    t.fit(X=np.array(list(zip(*idx))), weights=density)
    spans = t.diagram_
    spans = sorted(spans[:, 0] - spans[:, 1])
    idx_mid = int(len(spans) * idx_coef)
    tau = float(spans[idx_mid])

    # tau = coef * np.max(density)
    print(f'mid tau {tau}')
    t = Tomato(graph_type='radius',
               r=1.8,
               metric='euclidean',
               density_type='manual',
               merge_threshold=tau,
               )
    t.fit(X=np.array(list(zip(*idx))), weights=density)
    labels = t.labels_ + 1
    return labels


def load_pbc():
    pred = pd.read_csv('clusters.txt', header=None)
    pred = pred.fillna(0).to_numpy(dtype=np.int32).squeeze()
    return pred


def pbc(x, idx, f_pointcloud, radius, th, max_num_classes, scale_dim=None):
    """

    :param data_path: path to the tiff file
    :return: point_set_path: path to the point set defined by the user
    """

    subprocess.run(["./tomatoyue", f_pointcloud, radius, th])
    pred_tiff = np.zeros(x.shape, dtype=np.int32)
    pred = load_pbc()
    pred_tiff[idx] = pred
    if max_num_classes is not None:
        if np.max(pred_tiff) > max_num_classes:
            print('removing small clusters...')
            pred_tiff = remove_small_clusters(pred_tiff, max_num_clusters=max_num_classes)
    return pred_tiff


def iou(set_a, set_b):
    return len(set_a & set_b) / len(set_a.union(set_b))


def intersection(set_a, set_b):
    return len(set_a & set_b) / len(set_a)


def arrange_by_persistence(preds):
    pred_list = []

    while len(preds) > 0:
        pred_cur = preds[0]
        idx_delete_list = [0]
        pred_cell_list = [pred_cur]
        idx_set = set([i for i in range(0, len(preds))])
        for i in range(1, len(preds)):
            set_cur = set(zip(*pred_cur))
            set_i = set(zip(*preds[i]))
            if intersection(set_cur, set_i) > 0:
                pred_cell_list.append(preds[i])
                idx_delete_list.append(i)
        pred_list.append(pred_cell_list)
        idx_set = idx_set.difference(set(idx_delete_list))
        preds = [preds[i] for i in list(idx_set)]
    return pred_list


def check_non_overlap(pred, nuclei_coor_list):
    for i in range(len(nuclei_coor_list)):
        nuclei_coor = nuclei_coor_list[i]
        pred_i = pred[nuclei_coor]
        pred_i[np.where(pred_i != 0)] -= np.max(pred_i)
        if np.sum(pred_i) != 0:
            return False
    return True


def rearrange_idx_by_persi(x):
    """
    x: n_persi * z * x * y
    """
    x_r = []
    for p in range(len(x)):
        idx_list = np.unique(x[p]).tolist()[1:]
        np.random.shuffle(idx_list)
        temp = np.zeros(x[p].shape, dtype=np.uint16)
        for i in range(len(idx_list)):
            temp[np.where(x[p] == idx_list[i])] = i + 1
        x_r.append(temp)
    return np.array(x_r)


def rearrange_idx(x):
    idx_list = np.unique(x).tolist()[1:]
    np.random.shuffle(idx_list)
    temp = np.zeros(x.shape)
    for i in range(len(idx_list)):
        temp[np.where(x == idx_list[i])] = i + 1
    return temp


def separation_score(score_list):
    """

    :param score_list: n_cell * n_nuclei
    :return: best score and n_nuclei indices
    """

    max_score_list = []
    max_idx_list = []
    for i in range(len(score_list[0])):
        temp = np.array(score_list)
        target_col = np.expand_dims(temp[:, i], axis=1)
        rest_col = np.delete(temp, i, 1)
        score = target_col * np.prod(1 - rest_col, axis=1, keepdims=True)
        max_score_list.append(np.max(score))
        max_idx_list.append(np.argmax(score) + 1)
    return max_score_list, max_idx_list


def gen_critical_points(pred):
    idx_list = list(zip(*np.where(pred > 0)))
    cp = np.zeros_like(pred)
    padding = 3
    for idx in idx_list:
        ids = np.unique(pred[idx[0] - padding: idx[0] + padding, idx[1] - padding: idx[1] + padding,
                        idx[2] - padding: idx[2] + padding]).tolist()
        if len(ids) > 2 or (len(ids) == 2 and ids[0] > 0):
            cp[idx[0], idx[1], idx[2]] = 1
    return cp


def gen_idx_boundary(pred):
    return np.where(
        (pred[:-1, :, :] - pred[1:, :, :]) | (pred[:, :-1, :] - pred[:, 1:, :]) | (pred[:, :, :-1] - pred[:, :, 1:]))


def split_cluster_persi_boundary(fname, temp_dir, pred_cell_pbc, pred_cell_low_temp, split_id, nuclei_hit_list,
                                 merge_th):
    '''pred_pbc_low cropped and masked'''

    data_cell_otsu = imread(temp_dir + fname + '_data_cell_otsu.tiff')
    # pred_cell_low_temp = np.copy(pred_cell_low)
    pred_cell_low_temp[np.where(pred_cell_pbc != split_id)] = 0
    idx_split = np.where(pred_cell_pbc == split_id)
    density = data_cell_otsu[idx_split]
    # data_cell_otsu = None
    pred_cell_low_cropped, x_min, x_max, y_min, y_max, z_min, z_max = crop_by_idx(pred_cell_low_temp,
                                                                                  idx_split,
                                                                                  padding=10)

    n_nuclei = len(nuclei_hit_list)
    print(f'{len(nuclei_hit_list)} overlapped nuclei')
    reclustering = True
    while 1:
        dic_cell_cluster = {i: [] for i in range(n_nuclei)}
        overlap_idList_all = []
        visited = set()
        for i in range(n_nuclei):
            print(i)
            nuclei_mask_temp = np.zeros_like(pred_cell_pbc)
            nuclei_mask_temp[nuclei_hit_list[i]] = 1
            overlap_ids = np.unique(pred_cell_low_temp * nuclei_mask_temp)[1:]
            overlap_idList = []
            for overlap_id in overlap_ids:
                cluster = set(zip(*np.where(pred_cell_low_temp == overlap_id)))
                score_target = intersection(cluster, set(zip(*nuclei_hit_list[i])))
                score_other_sum = 0
                best_overlap_cell, best_score = i, score_target
                if score_target > 0:
                    for j in range(len(nuclei_hit_list)):
                        if j == i:
                            continue
                        score_other = intersection(cluster, set(zip(*nuclei_hit_list[j])))
                        if score_other > best_score:
                            best_overlap_cell = j
                            best_score = score_other
                        score_other_sum += score_other
                if score_target > 0 and score_other_sum == 0:
                    overlap_idList.append(overlap_id)
                else:
                    dic_cell_cluster[best_overlap_cell].append(overlap_id)
                visited.add(overlap_id)
            if not overlap_idList:
                if merge_th < 0.01:
                    print('perci_cell below re-clustering threshold, quit splitting ..')
                    reclustering = False
                    break
                reclustering = True
                merge_th /= 2
                print(f're-clustering p={merge_th}')
                t = Tomato(graph_type='radius',
                           r=1.8,
                           metric='euclidean',
                           density_type='manual',
                           merge_threshold=merge_th,
                           )
                t.fit(X=np.array(list(zip(*idx_split))), weights=density)
                pred_cell_low_temp = np.zeros(pred_cell_low_temp.shape)
                pred_cell_low_temp[idx_split] = t.labels_ + 1
                pred_cell_low_cropped, x_min, x_max, y_min, y_max, z_min, z_max = crop_by_idx(pred_cell_low_temp,
                                                                                              idx_split,
                                                                                              padding=10)
                break
            else:
                overlap_idList_all.append(overlap_idList)
                if i == len(nuclei_hit_list) - 1:
                    reclustering = False
        if not reclustering:
            break

    # pred_cell_low[idx_split] = pred_cell_low_temp[idx_split] + np.max(pred_cell_low)

    def gen_dic_adj_idx(pred_cell_low_cropped, cp_pad=3):
        print('generating dictionaries')
        dic_adj = defaultdict(set)
        dic_idx = defaultdict(None)
        for i in range(1, int(np.max(pred_cell_low_cropped)) + 1):
            idx = np.where(pred_cell_low_cropped == i)
            dic_idx[i] = idx
            idx_list = list(zip(*idx))
            for j in range(len(idx_list)):
                ids = np.unique(pred_cell_low_cropped[idx_list[j][0] - cp_pad: idx_list[j][0] + cp_pad,
                                idx_list[j][1] - cp_pad: idx_list[j][1] + cp_pad,
                                idx_list[j][2] - cp_pad: idx_list[j][2] + cp_pad]).tolist()
                if len(ids) < 2 or (len(ids) == 2 and ids[0] == 0):
                    continue
                else:
                    for id in ids:
                        if id != 0 and id != i:
                            dic_adj[i].add(id)
        return dic_adj, dic_idx

    for i in range(n_nuclei):
        dic_cell_cluster[i] += overlap_idList_all[i]
    # print(np.unique(pred_cell_low_cropped))
    dic_adj, dic_idx = gen_dic_adj_idx(pred_cell_low_cropped)
    print(len(np.unique(pred_cell_low_cropped)))
    print(len(dic_adj))
    print(len(dic_idx))

    print('merging clustering')
    while 1:
        print(len(visited), len(np.unique(pred_cell_low_cropped)))
        new_overlap_idList_all = []
        for i in range(len(overlap_idList_all)):
            new_idList = []
            for j in range(len(overlap_idList_all[i])):
                id = overlap_idList_all[i][j]
                for item in dic_adj[id]:
                    if item not in visited:
                        new_idList.append(item)
                        visited.add(item)
            new_overlap_idList_all.append(new_idList)
            dic_cell_cluster[i] += new_idList
        if sum([len(idList) for idList in new_overlap_idList_all]) == 0:
            break
        overlap_idList_all = new_overlap_idList_all

    pred_cell_split = np.zeros_like(pred_cell_pbc)
    for i in range(len(dic_cell_cluster)):
        pred_cell_split_temp = np.zeros(pred_cell_low_cropped.shape, dtype=np.int32)
        for id in dic_cell_cluster[i]:
            pred_cell_split_temp[dic_idx[id]] = i + 1
        pred_cell_split[z_min:z_max, x_min:x_max, y_min:y_max] += pred_cell_split_temp
    return pred_cell_split


def gen_nuclei_mask(nuclei_cc):
    mask = np.zeros(nuclei_cc.shape)
    for i in range(int(np.max(nuclei_cc))):
        idx = np.where(nuclei_cc == i)
        x_max = np.max(idx[1]) + 20
        x_min = np.min(idx[1]) - 20
        y_max = np.max(idx[2]) + 20
        y_min = np.min(idx[2]) - 20
        z_max = np.max(idx[0]) + 10
        z_min = np.min(idx[0]) - 10
        mask[z_min:z_max, x_min:x_max, y_min:y_max] = 1
    return mask


def otsu_interactive(x, v, save_dir, fname):
    # if os.path.exists('gui_config_otsu.txt'):
    #     with open('gui_config_otsu.txt') as f:
    #         v = float(f.readline())

    while 1:
        print('>>> cleaning cells, Otsu threshold = %s ' % str(v))
        x_copy = np.copy(x)
        x_copy = preprocessing(x_copy, invert=False, scale=True, otsu_coef=v)
        x_copy_2d = np.max(x_copy, axis=0)
        x_copy_2d[np.where(x_copy_2d > 0)] = 255

        x_img = np.uint8(x_copy_2d)

        # img = Image.fromarray(x_img, 'L')
        # save_path = save_dir + fname + 'otsu_cell_' + str(v) + '.png'
        # img.save(save_path)
        # x_img = np.uint8(x_copy_2d)
        # with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(x_img, name='Microglia')
        otsuPanel = OtsuPanelCell(v, str(len(np.where(x_copy > 0)[0])), viewer)
        exitPanel = ButtonExit()
        viewer.window.add_dock_widget(otsuPanel, name='OTSU CELL')
        viewer.window.add_dock_widget(exitPanel, name='EXIT')
        napari.run()

        if otsuPanel.accept:
            # with open('gui_config_otsu.txt', 'w') as f:
            #             #     f.write(str(v))
            return x_copy, x_img
        else:
            v = float(otsuPanel.sliderPanel.otsu)


def otsu_interactive_nuclei(x, cell_img, v, save_dir, fname):
    while 1:
        print('>>> cleaning nuclei, Otsu threshold = %s ' % str(v))
        x_copy = np.copy(x)
        x_copy = preprocessing(x_copy, invert=False, scale=False, otsu_coef=v)
        x_copy_2d = np.max(x_copy, axis=0)
        x_copy_2d[np.where(x_copy_2d > 0)] = 255
        x_img = np.uint8(x_copy_2d)
        viewer = napari.Viewer()
        viewer.add_image(cell_img, name='Microglia')
        viewer.add_image(x_img, name='Nuclei')
        otsuPanel = OtsuPanelNuclei(v, str(len(np.where(x_copy > 0)[0])), viewer)
        exitPanel = ButtonExit()
        viewer.window.add_dock_widget(otsuPanel, name='OTSU NUCLEI')
        viewer.window.add_dock_widget(exitPanel, name='EXIT')
        napari.run()

        if otsuPanel.accept:
            return x_copy, x_img, True
        elif otsuPanel.back:
            return x_copy, x_img, False
        else:
            v = float(otsuPanel.sliderPanel.otsu)


def colorization(img, save_path):
    color_dic = mcolors.XKCD_COLORS

    num_masks = int(np.max(img))

    keys_all = [k for k in color_dic.keys() if k != 'black']
    perm = np.random.permutation(len(keys_all))[:num_masks]
    keys = [keys_all[i] for i in perm]

    print('%d masks' % num_masks)
    mask = np.zeros(list(img.shape) + [3])
    for i in range(1, num_masks + 1):
        vec = mcolors.to_rgb(color_dic[keys[i - 1]])
        mask[np.where(img == i)] = vec

    mask = np.uint8(mask * 255)
    img = Image.fromarray(mask, 'RGB')
    img.save(save_path)


def visualize_nuclei_cp(fname, cellpose_dir, data_cell, data_nuclei, z_len, n_batches, v, cell_img, nuclei_img_list):
    pred_ch_list = []
    for i in range(n_batches):
        pred_ch = imread(f'{cellpose_dir}{fname}_ch{i}_cp_masks.tif')
        pred_ch_list.append(pred_ch)

    nuclei_cp_idx = 0
    last_cell_idx = 0
    while 1:
        if nuclei_cp_idx == -1:
            pred_nuclei_3D = np.zeros(data_nuclei.shape, dtype=np.uint16)
            for i in range(len(pred_ch_list)):
                pred_ch = pred_ch_list[i]
                pred_ch += np.max(pred_nuclei_3D)
                pred_ch[np.where(pred_ch == np.max(pred_nuclei_3D))] = 0
                pred_nuclei_3D[i * z_len: (i + 1) * z_len] = to3D(pred_ch, data_nuclei[i * z_len: (i + 1) * z_len])
            viewer = napari.Viewer(title=f'All Nuclei', ndisplay=3)
            layer_nuclei = viewer.add_labels(pred_nuclei_3D)

            @layer_nuclei.bind_key(key='b')
            def back(layer_nuclei):
                nonlocal nuclei_cp_idx, last_cell_idx
                nuclei_cp_idx = last_cell_idx
                viewer.close()

            napari.run()

        else:
            viewer = napari.Viewer(title=f'Cellpose Prediction {nuclei_cp_idx + 1}/{n_batches}')
            viewer.add_image(cell_img, name='Microglia')
            layer_nuclei_data = viewer.add_image(nuclei_img_list[nuclei_cp_idx], name='Nuclei')
            layer_cell_data = viewer.add_image(np.max(data_cell[nuclei_cp_idx * z_len: (nuclei_cp_idx + 1) * z_len], axis=0))
            layer_nuclei = viewer.add_labels(pred_ch_list[nuclei_cp_idx])
            instructPanel = CellposeInstruction()
            maskPanel = NucleiMaskPanel(v, viewer)
            exitPanel = ButtonExit()
            viewer.window.add_dock_widget(instructPanel, name='Instructions')
            viewer.window.add_dock_widget(maskPanel, name='Commands')
            viewer.window.add_dock_widget(exitPanel, name='Exit')
            history = []

            @layer_nuclei.mouse_double_click_callbacks.append
            def on_second_click_of_double_click(layer_nuclei, event):
                cluster_id = layer_nuclei.data[int(event.position[0]), int(event.position[1])]
                data_nuclei_masks = layer_nuclei.data
                if cluster_id != 0:
                    """remove nuclei"""
                    idx = np.where(data_nuclei_masks == cluster_id)
                    data_nuclei_masks[idx] = 0
                    history.append(('d', cluster_id, idx))
                else:
                    """add nuclei"""
                    half_width = v / 4
                    pos = event.position
                    cluster_id = np.max(data_nuclei_masks) + 1
                    data_nuclei_masks[int(pos[0] - half_width): int(pos[0] + half_width),
                    int(pos[1] - half_width): int(pos[1] + half_width)] = cluster_id
                    print('cluster %d added' % cluster_id)
                    idx = np.where(data_nuclei_masks == cluster_id)
                    history.append(('a', cluster_id, idx))
                layer_nuclei.data = data_nuclei_masks
                pred_ch_list[nuclei_cp_idx] = data_nuclei_masks

            @layer_nuclei.bind_key(key='a')
            def show_all(layer_nuclei):
                nonlocal nuclei_cp_idx
                nuclei_cp_idx = -1
                viewer.close()

            @layer_nuclei.bind_key('u')
            def undo_remove(layer_nuclei):
                if len(history) == 0:
                    print('No action')
                    return
                action_type, cluster_id, idx = history.pop()
                pred_cur = layer_nuclei.data
                if action_type == 'd':
                    pred_cur[idx] = cluster_id
                    layer_nuclei.data = pred_cur
                    print('Undo cluster %d' % cluster_id)
                elif action_type == 'a':
                    pred_cur[idx] = 0
                    layer_nuclei.data = pred_cur
                    print('Undo cluster %d' % cluster_id)
                pred_ch_list[nuclei_cp_idx] = pred_cur

            @layer_nuclei.bind_key(key='Right')
            def next_cell(layer_nuclei):
                nonlocal nuclei_cp_idx, last_cell_idx
                if nuclei_cp_idx < n_batches - 1:
                    nuclei_cp_idx += 1
                else:
                    nuclei_cp_idx = 0
                last_cell_idx = nuclei_cp_idx

                layer_nuclei_data.data = nuclei_img_list[nuclei_cp_idx]
                layer_cell_data.data = np.max(data_cell[nuclei_cp_idx * z_len: (nuclei_cp_idx + 1) * z_len], axis=0)
                layer_nuclei.data = pred_ch_list[nuclei_cp_idx]
                viewer.title = f'Cellpose Prediction {nuclei_cp_idx + 1}/{n_batches}'

            @layer_nuclei.bind_key(key='Left')
            def prev_cell(layer_nuclei):
                nonlocal nuclei_cp_idx, last_cell_idx
                if nuclei_cp_idx > 0:
                    nuclei_cp_idx -= 1
                else:
                    nuclei_cp_idx = n_batches - 1
                last_cell_idx = nuclei_cp_idx

                layer_nuclei_data.data = nuclei_img_list[nuclei_cp_idx]
                layer_cell_data.data = np.max(data_cell[nuclei_cp_idx * z_len: (nuclei_cp_idx + 1) * z_len], axis=0)
                layer_nuclei.data = pred_ch_list[nuclei_cp_idx]
                viewer.title = f'Cellpose Prediction {nuclei_cp_idx + 1}/{n_batches}'


            napari.run()
            if maskPanel.state == 2:
                return None, 2, None
            if maskPanel.state != -1:
                return pred_ch_list, maskPanel.state, maskPanel.lineEdit.getValue()

        # print(maskPanel.state)

        # @layer_nuclei.bind_key(key='Escape')
        # def terminate(layer_label):
        #     viewer.close()
        #     nonlocal Flag_exit
        #     Flag_exit = True

        # if maskPanel.accept:
        #     return


def z_stitching(pred_nuclei_3D):
    pad = 3
    ids = np.unique(pred_nuclei_3D)[1:]
    dic_parent = {id: id for id in ids}
    for id in ids:
        idx = np.where(pred_nuclei_3D == id)
        for i in range(len(idx[0])):
            z_start, z_end = max(0, idx[0][i] - pad), min(pred_nuclei_3D.shape[0], idx[0][i] + pad)
            ids_neighbor = np.unique(pred_nuclei_3D[z_start:z_end, idx[1][i], idx[2][i]])
            if len(ids) < 2 or (len(ids) == 2 and ids[0] == 0):
                continue
            else:
                for id_n in ids_neighbor:
                    if id_n != 0 and id_n != id and id_n != dic_parent[id]:
                        dic_parent[id_n] = id

    def find_parent(id):
        if dic_parent[id] == id:
            return id
        else:
            return find_parent(dic_parent[id])

    print(dic_parent)
    for id in dic_parent:
        id_parent = find_parent(id)
        if id_parent != id:
            pred_nuclei_3D[pred_nuclei_3D == id] = id_parent
    return pred_nuclei_3D


def cellpose_interactive(data_nuclei, data_cell, cellpose_dir, fname, interactive_dir, v, z_len, cell_img, nuclei_img):
    """
    img_nuclei: an input image file to Cellpose
    img_cell: 2d cell for visualization
    """
    print('***interactive commands***\n'
          '\n options a: accept b: try a new diameter')
    while 1:
        print('>>> nuclei detection with Cellpose, cell diameter = %s ' % str(v))
        flist = os.listdir(cellpose_dir)
        for f in flist:
            os.remove(cellpose_dir + f)

        start_idx = 0
        n_batches = int(np.floor((len(data_nuclei) / z_len)))
        print(f'{n_batches} z-batches')
        nuclei_img_list = []
        for i in range(n_batches):
            x = np.max(data_nuclei[start_idx:start_idx + z_len, :, :], axis=0)
            x[np.where(x > 0)] = 255
            nuclei_img_list.append(x)
            imwrite(f'{cellpose_dir}{fname}_ch{i}.tif', x)
            start_idx += z_len

        subprocess.run(['python', '-m', 'cellpose', '--dir', 'cellpose/',
                        '--pretrained_model', 'nuclei', '--diameter', str(v), '--save_tif', '--fast_mode'])

        pred_ch_list, state, v = visualize_nuclei_cp(fname, cellpose_dir, data_cell, data_nuclei, z_len, n_batches, v,
                                                     cell_img, nuclei_img_list)


        # print('***interactive commands***\n'
        #       '\n options a: accept b: try a new diameter')
        # arg = input()
        if state == 1:
            print('Saving Nuclei Prediction ...')
            pred_nuclei_3D = np.zeros(data_nuclei.shape, dtype=np.uint16)
            for i in range(len(pred_ch_list)):
                pred_ch = pred_ch_list[i]
                pred_ch += np.max(pred_nuclei_3D)
                pred_ch[np.where(pred_ch == np.max(pred_nuclei_3D))] = 0
                pred_nuclei_3D[i * z_len: (i + 1) * z_len] = to3D(pred_ch, data_nuclei[i * z_len: (i + 1) * z_len])

            pred_nuclei_3D = z_stitching(pred_nuclei_3D)
            # with napari.gui_qt():
            #     viewer = napari.Viewer(title=f'Nuclei Stitching', ndisplay=3)
            #     layer_nuclei = viewer.add_labels(pred_nuclei_3D)

            print('Nuclei prediction saved')
            print('%d nuclei' % int(len(np.unique(pred_nuclei_3D)) - 1))
            return pred_nuclei_3D, True
        elif state == 2:
            return None, False


def to3D(nuclei_2d_mask, nuclei_3d_otsu_mask):
    temp_3d = np.tile(nuclei_2d_mask, [nuclei_3d_otsu_mask.shape[0], 1, 1])
    temp_3d[np.where(nuclei_3d_otsu_mask == 0)] = 0
    return temp_3d


def remove_repeated_cc(pred_nuclei_3D_cc, pred_nuclei_2D):
    pred_nuclei_2D_cc = np.max(pred_nuclei_3D_cc, axis=0)
    id_cc_list = np.unique(pred_nuclei_2D_cc)[1:]
    dic = {}
    pred_temp = np.copy(pred_nuclei_2D)
    pred_temp[np.where(pred_temp > 0)] = 1
    for id in id_cc_list:
        temp = np.copy(pred_nuclei_2D_cc)
        temp[np.where(temp != id)] = 0
        temp[np.where(temp == id)] = 1
        id_nuclei = np.unique(temp * pred_nuclei_2D)[1]
        size = np.sum(pred_temp * temp)
        if id_nuclei not in dic:
            dic[id_nuclei] = [id, size]
        else:
            size_curr = dic[id_nuclei][1]
            if size > size_curr:
                dic[id_nuclei] = [id, size]

    pred_removed = np.zeros(pred_nuclei_3D_cc.shape)
    id_cnt = 1
    for key in dic:
        pred_removed[np.where(pred_nuclei_3D_cc == dic[key][0])] = id_cnt
        id_cnt += 1
    return pred_removed


def remove_by_cellpose(pred_nuclei_cellpose, pred_nuclei_pbc_cc):
    """
    adding nuclei detected by Cellpose to pbc_cc predictions
    """
    # id_set = set([i for i in range(1, np.max(pred_nuclei_cellpose) + 1)])
    id_set = set(np.unique(pred_nuclei_cellpose)[1:])
    temp = np.copy(pred_nuclei_cellpose)
    temp[np.where(temp > 1)] = 1
    id_overlap = np.unique(temp * pred_nuclei_pbc_cc)
    id_nuclei_pbc_cc = np.unique(pred_nuclei_pbc_cc)[1:]
    for id in id_nuclei_pbc_cc:
        if id not in id_overlap:
            pred_nuclei_pbc_cc[np.where(pred_nuclei_pbc_cc == id)] = 0
    temp = np.copy(pred_nuclei_pbc_cc)
    temp[np.where(temp > 1)] = 1
    id_set = id_set.difference(set(np.unique(temp * pred_nuclei_cellpose)))
    max_pbc_id = np.max(pred_nuclei_pbc_cc) + 1
    for e in id_set:
        pred_nuclei_pbc_cc[np.where(pred_nuclei_cellpose == e)] = max_pbc_id
        max_pbc_id += 1
    return pred_nuclei_pbc_cc


# def merge_nuclei_pred(pred_nuclei_cellpose, pred_nuclei_pbc_cc):
#     """
#     adding nuclei detected by Cellpose to pbc_cc predictions
#     """
#     id_set = set([i for i in range(1, np.max(pred_nuclei_cellpose) + 1)])
#     temp = np.copy(pred_nuclei_cellpose)
#     temp[np.where(temp > 1)] = 1
#     id_overlap = np.unique(temp * pred_nuclei_pbc_cc)
#     id_nuclei_pbc_cc = np.unique(pred_nuclei_pbc_cc)[1:]
#     for id in id_nuclei_pbc_cc:
#         if id not in id_overlap:
#             pred_nuclei_pbc_cc[np.where(pred_nuclei_pbc_cc == id)] = 0
#     temp = np.copy(pred_nuclei_pbc_cc)
#     temp[np.where(temp > 1)] = 1
#     id_set = id_set.difference(set(np.unique(temp * pred_nuclei_cellpose)))
#     max_pbc_id = np.max(pred_nuclei_pbc_cc) + 1
#     for e in id_set:
#         pred_nuclei_pbc_cc[np.where(pred_nuclei_cellpose == e)] = max_pbc_id
#         max_pbc_id += 1
#     return pred_nuclei_pbc_cc


def check_nuclei_fast(pred_nuclei, pred_pbc):
    pred_nuclei[np.where(pred_nuclei > 1)] = 1
    id_overlap_list = np.unique(pred_nuclei * pred_pbc)[1:]
    pred_copy = np.zeros(pred_pbc.shape)
    for i in range(min(len(id_overlap_list), 100)):
        print(i, len(id_overlap_list))
        pred_copy[np.where(pred_pbc == id_overlap_list[i])] = i + 1
    return pred_copy


def crop(data):
    idx = np.array(list(zip(*np.where(data > 0))))
    x_min = np.min(idx[:, 1])
    x_max = np.max(idx[:, 1])
    y_min = np.min(idx[:, 2])
    y_max = np.max(idx[:, 2])
    z_min = np.min(idx[:, 0])
    z_max = np.max(idx[:, 0])
    data = data[z_min:z_max, x_min:x_max, y_min:y_max]
    return data


def crop_by_idx(pred, idx, padding=100):
    """
    data: 4-dim data with z, nuclei/microgli, x, y
    """
    # print(idx)
    idx = np.array(list(zip(*idx)))
    x_min = max(np.min(idx[:, 1]) - padding, 0)
    x_max = min(np.max(idx[:, 1]) + padding, pred.shape[1])
    y_min = max(np.min(idx[:, 2]) - padding, 0)
    y_max = min(np.max(idx[:, 2]) + padding, pred.shape[2])
    z_min = max(np.min(idx[:, 0]) - padding, 0)
    z_max = min(np.max(idx[:, 0]) + padding, pred.shape[0])
    return pred[z_min:z_max, x_min:x_max, y_min:y_max], x_min, x_max, y_min, y_max, z_min, z_max


def create_ball(boundary):
    idx = list(zip(*np.where(boundary > 0)))
    boundary_new = np.zeros(boundary.shape)
    boundary_shape = boundary.shape
    # print(boundary_new.shape)
    # print(boundary_new.shape)
    for i in range(len(idx)):
        x_min = max(idx[i][0] - 2, 0)
        x_max = min(idx[i][0] + 2, boundary_shape[0])
        y_min = max(idx[i][1] - 2, 0)
        y_max = min(idx[i][1] + 2, boundary_shape[1])
        z_min = max(idx[i][2] - 2, 0)
        z_max = min(idx[i][2] + 2, boundary_shape[2])
        boundary_new[x_min:x_max, y_min:y_max, z_min:z_max] = boundary[idx[i][0], idx[i][1], idx[i][2]]
    return boundary_new


def gen_boundary_fast_v2(pred):
    boundary = np.zeros(pred.shape)

    pred = ndimage.maximum_filter(pred, size=(3, 3, 3))
    filter_min = ndimage.minimum_filter(pred, size=(3, 3, 3))
    filter_max = ndimage.maximum_filter(pred, size=(3, 3, 3))
    filter_median = ndimage.median_filter(pred, size=(3, 3, 3))

    # pred = ndimage.maximum_filter(pred, size=(4, 4, 4))
    # filter_min = ndimage.minimum_filter(pred, size=(4, 4, 4))
    # filter_max = ndimage.maximum_filter(pred,size=(4, 4, 4))
    # filter_median = ndimage.median_filter(pred, size=(4, 4, 4))

    boundary[np.where((filter_max != filter_median) & (filter_min == 0) & (filter_median != 0))] = 255
    boundary[np.where((filter_max != filter_min) & (filter_min != 0))] = 255

    return boundary


def visualize_pred_napari(pred):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_labels(pred)


def gen_boundary_fast(pred):
    # boundary = np.zeros(pred.shape)
    # id_list = np.unique(pred)[1:]
    # for i in range(2, int(np.max(pred)) + 1):

    th = np.max(pred)
    pred[np.where(pred == 0)] = 2 * th + 10

    shift_x = np.copy(pred)
    shift_x = shift_x[:, 1:, :]
    pad = np.ones([shift_x.shape[0], 1, shift_x.shape[2]]) * 2 * th + 10
    shift_x = np.concatenate((shift_x, pad), axis=1)

    shift_y = np.copy(pred)
    shift_y = shift_y[:, :, 1:]
    pad = np.ones([shift_y.shape[0], shift_y.shape[1], 1]) * 2 * th + 10
    shift_y = np.concatenate((shift_y, pad), axis=2)

    shift_z = np.copy(pred)
    shift_z = shift_z[1:, :, :]
    pad = np.ones([1, shift_z.shape[1], shift_z.shape[2]]) * 2 * th + 10
    shift_z = np.concatenate((shift_z, pad), axis=0)

    boundary_x = np.abs(pred - shift_x)
    boundary_x[np.where(boundary_x > th + 5)] = 0
    boundary_y = np.abs(pred - shift_y)
    boundary_y[np.where(boundary_y > th + 5)] = 0
    boundary_z = np.abs(pred - shift_z)
    boundary_z[np.where(boundary_z > th + 5)] = 0

    boundary = boundary_x + boundary_y + boundary_z

    boundary[np.where(boundary > 0)] = 255

    boundary = create_ball(boundary)

    return boundary


def gen_boundary(pred):
    # data = imread('/home/yue/Desktop/data/Microglia_December_2020/20201105-slide7-5-4_slice2_right.tif')
    # data = data[:, 1, :, :]
    # imwrite('test_data_original.tiff', data)
    # data = preprocessing(data, invert=False, scale=True, otsu_coef=0.5)
    # a = imread(
    #     '/media/yue/Data/PycharmProjects_Backup/microglia/temp/20201105-slide7-5-4_slice2_right.tif_pbc_0.7.tiff')
    #
    # data[np.where(a != 34)] = 0
    # data = crop(data)
    # idx = list(zip(*np.where(data > 0)))
    #
    # imwrite('test_data.tiff', data)
    # imwrite('test_mask.tiff', a)
    # pred = pbc(x=data,
    #            f_pointcloud='pointcloud.txt',
    #            radius='1,8',
    #            th='0.5',
    #            max_num_classes=None)
    # imwrite('pbc_temp.tiff', pred)

    # pred = imread('pbc_temp.tiff')
    boundary = np.zeros(pred.shape)
    id_list = np.unique(pred)[1:]
    # for i in range(2, int(np.max(pred)) + 1):
    for i in id_list:
        print('%d / %d' % (i, int(np.max(pred))))
        # for i in range(2, 5):
        pred_temp = np.copy(pred)
        pred_temp[np.where((pred_temp != i) & (pred_temp != 0))] = i + 1
        pred_temp[np.where(pred_temp == 0)] = -100
        # pred_temp[np.where(pred_temp == i)] = 1
        # imwrite('pred_temp.tiff', pred_temp)

        shift_x = np.copy(pred_temp)
        shift_x = shift_x[:, 1:, :]
        pad = np.zeros([shift_x.shape[0], 1, shift_x.shape[2]])
        shift_x = np.concatenate((shift_x, pad), axis=1)

        shift_y = np.copy(pred_temp)
        shift_y = shift_y[:, :, 1:]
        pad = np.zeros([shift_y.shape[0], shift_y.shape[1], 1])
        shift_y = np.concatenate((shift_y, pad), axis=2)

        shift_z = np.copy(pred_temp)
        shift_z = shift_z[1:, :, :]
        pad = np.zeros([1, shift_z.shape[1], shift_z.shape[2]])
        shift_z = np.concatenate((shift_z, pad), axis=0)

        # print(np.unique(np.abs(pred_temp - shift_x)))
        # print(np.unique(np.abs(pred_temp - shift_y)))
        # print(np.unique(np.abs(pred_temp - shift_z)))
        # print('+=====')

        boundary_x = np.abs(pred_temp - shift_x)
        boundary_x[np.where(boundary_x != 1)] = 0
        boundary_y = np.abs(pred_temp - shift_y)
        boundary_y[np.where(boundary_y != 1)] = 0
        boundary_z = np.abs(pred_temp - shift_z)
        boundary_z[np.where(boundary_z != 1)] = 0

        boundary_temp = boundary_x + boundary_y + boundary_z

        boundary += boundary_temp

        # imwrite('test_boundary_temp.tiff', boundary_x)
    boundary = create_ball(boundary)
    # boundary[np.where(boundary >= 1)] = 1
    return boundary
    # imwrite('test_boundary.tiff', boundary)


def create_critical_points(pred_cell, pred_low, pred_nuclei_cellpose):
    """
    pred: largest prediction of a cell
    """
    # max_label = int(np.max(pred))
    # pred_boundary = np.zeros(pred.shape[1:])
    # id_list = np.unique(pred)[1:]

    # for i in id_list:
    # pred_cell = pred[i]
    pred_boundary = np.zeros(pred_low.shape)
    pred_low_temp = pred_low
    pred_low_temp[np.where(pred_cell == 0)] = 0
    # pred_low_temp[np.where(pred_nuclei_cellpose > 0)] = 0
    # pred_low_temp = crop(pred_low_temp)
    idx = np.array(list(zip(*np.where(pred_low_temp > 0))))
    # print(idx.shape)
    x_min = np.min(idx[:, 1])
    x_max = np.max(idx[:, 1])
    y_min = np.min(idx[:, 2])
    y_max = np.max(idx[:, 2])
    z_min = np.min(idx[:, 0])
    z_max = np.max(idx[:, 0])
    pred_low_temp = pred_low_temp[z_min:z_max, x_min:x_max, y_min:y_max]

    boundary = gen_boundary_fast_v2(pred_low_temp.astype(np.int32))
    # boundary = gen_boundary_fast(pred_low_temp.astype(np.int32))
    boundary[np.where(pred_nuclei_cellpose[z_min:z_max, x_min:x_max, y_min:y_max] > 0)] = 0

    pred_boundary[z_min:z_max, x_min:x_max, y_min:y_max] = boundary

    return pred_boundary


def pipeline_setup(f_data, interactive_dir, cellpose_dir, save_dir):
    """
    No modification to data
    """
    data = imread(f_data)
    fname = f_data.split('/')[-1]

    print('processing %s' % str(fname))
    print('Input dim. %s' % str(data.shape))

    if not os.path.exists(interactive_dir):
        os.mkdir(interactive_dir)

    if not os.path.exists(cellpose_dir):
        os.mkdir(cellpose_dir)

    shape = list([data.shape[0]]) + list(data.shape[2:])
    f_pred_dir = os.path.join(save_dir, fname)
    return data, fname, f_pred_dir, shape


# def pipeline_setup(f_data, interactive_dir, cellpose_dir, save_dir, z_len):
#     data = imread(f_data)
#     fname = f_data.split('/')[-1]
#
#     print('processing %s' % str(fname))
#     print('Input dim. %s' % str(data.shape))
#
#     if not os.path.exists(fname):
#         os.mkdir(fname)
#
#     if not os.path.exists(interactive_dir):
#         os.mkdir(interactive_dir)
#
#     if not os.path.exists(cellpose_dir):
#         os.mkdir(cellpose_dir)
#
#     num_files = len(data) // z_len
#     for i in range(num_files):
#         data_batch = data[i * z_len: (i + 1) * z_len]
#         imwrite(f'{f_data}/batch_{i}', data_batch)
#
#     shape = list([z_len]) + list(data.shape[2:])
#     f_pred_dir = os.path.join(save_dir, fname)
#     return fname, f_pred_dir, shape


def pipeline_preprocessing(fname, data, interactive_dir, cellpose_dir, temp_dir, otsu_cell, otsu_nuclei,
                           cellpose_diameter, z_len):
    fname_cp_noext = fname.replace('.tif', '').replace('.tiff', '')
    step = 0
    data_cell = np.copy(data[:, 1, :, :])
    while 1:
        if step == 0:
            data_nuclei = np.copy(data[:, 0, :, :])
            data_cell_otsu, cell_img = otsu_interactive(data_cell, otsu_cell, interactive_dir, fname_cp_noext)
            data_nuclei[np.where(data_cell_otsu == 0)] = 0
            step = 1
        if step == 1:
            data_nuclei, nuclei_img, accept = otsu_interactive_nuclei(data_nuclei, cell_img, otsu_nuclei,
                                                                      interactive_dir, fname_cp_noext)
            if accept:
                step = 2
            else:
                step = 0
        if step == 2:
            pred_nuclei_cellpose, accept = cellpose_interactive(data_nuclei, data_cell, cellpose_dir, fname_cp_noext,
                                                                interactive_dir,
                                                                cellpose_diameter, z_len, cell_img, nuclei_img)
            if accept:
                break
            else:
                step = 1

    imwrite(temp_dir + fname + '_nuclei_mask_3d.tiff', pred_nuclei_cellpose)
    imwrite(temp_dir + fname + '_data_cell_otsu.tiff', data_cell_otsu)


# def pipeline_preprocessing(fname, data, interactive_dir, cellpose_dir, temp_dir, otsu_cell,
#                            otsu_nuclei, cellpose_diameter, ds, radius_nuclei, persi_nuclei, nuclei_size_th, scale_dim):
#
#     fname_cp_noext = fname.replace('.tif', '').replace('.tiff', '')
#     data_cell = data[:, 1, :, :]
#     data_nuclei = data[:, 0, :, :]
#     data_celL_otsu_2d, data_cell_otsu = otsu_interactive(data_cell, otsu_cell, interactive_dir, fname_cp_noext)
#     data_nuclei[np.where(data_cell_otsu == 0)] = 0
#     img_nuclei, otsu_val_nuclei = otsu_interactive_nuclei(data_nuclei, otsu_nuclei, interactive_dir, fname_cp_noext)
#     pred_nuclei_2D = cellpose_interactive(img_nuclei, data_celL_otsu_2d, cellpose_dir, fname_cp_noext, interactive_dir, cellpose_diameter)
#     data_nuclei[np.where(data_nuclei < otsu_val_nuclei)] = 0
#     pred_nuclei_3D = to3D(pred_nuclei_2D, data_nuclei)
#
#     pred_nuclei_3D_cc = cc3d.connected_components(np.int8(pred_nuclei_3D))
#     pred_nuclei_3D_cc = pick_top_k_clusters(pred_nuclei_3D_cc, k=int(np.max(pred_nuclei_2D)))
#     pred_nuclei_3D_cc = remove_repeated_cc(pred_nuclei_3D_cc, pred_nuclei_2D)
#     pred_nuclei_3D_cc[np.where(pred_nuclei_3D_cc > 0)] = 1
#     pred_nuclei_cellpose = cc3d.connected_components(np.int8(pred_nuclei_3D_cc))
#
#     print('*** PBC nuclei ***')
#     pred_nuclei_pbc = pbc(x=data_nuclei,
#                           idx=np.where(data_nuclei != 0),
#                           f_pointcloud=fname + 'pointcloud.txt',
#                           radius=radius_nuclei,
#                           th=persi_nuclei,
#                           max_num_classes=None)
#
#     pred_nuclei_pbc[np.where(pred_nuclei_pbc > 0)] = 1
#     pred_nuclei_pbc_cc = cc3d.connected_components(np.int8(pred_nuclei_pbc))
#     pred_nuclei_pbc_cc = remove_by_size(pred_nuclei_pbc_cc, nuclei_size_th)
#     imwrite(temp_dir + fname + '_test_nuclei_pbc_cc.tif', pred_nuclei_pbc_cc)
#     pred_nuclei_cellpose = merge_nuclei_pred(pred_nuclei_cellpose, pred_nuclei_pbc_cc)
#
#     imwrite(temp_dir + fname + '_nuclei_mask_3d.tiff', pred_nuclei_cellpose)
#     imwrite(temp_dir + fname + '_data_cell_otsu.tiff', data_cell_otsu)


def p_binary_search(p, density, labels):
    p_min = 0.9 * p
    start, end = 0, np.max(labels)
    best_id, best_dist = None, 1.1
    while start <= end:
        target = start + (end - start) // 2
        density_cluster = density[labels == target]
        p_cluster = max(density_cluster) - min(density_cluster)
        print((target, p_cluster))
        if p_min <= p_cluster <= p:
            return target
        if p_cluster < p_min:
            if p_min - p_cluster < best_dist:
                best_id = target
                best_dist = p_min - p_cluster
            end = target - 1
        if p_cluster > p:
            if p_cluster - p < best_dist:
                best_id = target
                best_dist = p_cluster - p
            start = target + 1
    return best_id


def pipeline_cell_pbc_gudhi(fname, persi_cell_list, radius_cell, temp_dir):
    print('*** PBC cell ***')
    data_cell_otsu = imread(temp_dir + fname + '_data_cell_otsu.tiff')
    # f_pointcloud = 'pointcloud.txt'
    idx = np.where(data_cell_otsu != 0)
    density = data_cell_otsu[idx]
    x = np.array(list(zip(*idx)))

    for p in persi_cell_list:
        file_path = temp_dir + fname + '_pbc_' + p + '.tiff'
        print('p_cell=%s' % p)
        t = Tomato(graph_type='radius',
                   r=float(radius_cell),
                   metric='euclidean',
                   density_type='manual',
                   merge_threshold=float(p),
                   )
        t.fit(X=x, weights=density)
        target = p_binary_search(float(p), density, t.labels_)
        # target = 1000
        print(f'target={target}')
        pred_cell_pbc = np.zeros(data_cell_otsu.shape, dtype=np.int32)
        pred_cell_pbc[idx] = t.labels_ + 1
        imwrite(file_path + 'original.tiff', pred_cell_pbc)
        pred_cell_pbc[pred_cell_pbc > target] = 0
        imwrite(file_path, pred_cell_pbc)


# def pipeline_cell_pbc(fname, persi_cell_list,
#                       radius_cell, temp_dir, scale_dim):
#     print('*** PBC cell ***')
#     data_cell_otsu = imread(temp_dir + fname + '_data_cell_otsu.tiff')
#     f_pointcloud = 'pointcloud.txt'
#     idx = np.where(data_cell_otsu != 0)
#     with open(f_pointcloud, 'w') as f:
#         for i in range(len(idx[0])):
#             line = str(idx[0][i]) + " " + str(idx[1][i]) + " " + str(idx[2][i]) + " " + str(
#                 data_cell_otsu[idx[0][i], idx[1][i], idx[2][i]]) + "\n"
#             f.write(line)
#     print('pointcloud file generated! %d points' % len(idx[0]))
#     for p in persi_cell_list:
#         file_path = temp_dir + fname + '_pbc_' + p + '.tiff'
#         print('p_cell=%s' % p)
#         pred_cell_pbc = pbc(x=data_cell_otsu,
#                             idx=idx,
#                             f_pointcloud=f_pointcloud,
#                             radius=radius_cell,
#                             th=p,
#                             max_num_classes=None,
#                             scale_dim=scale_dim)
#         imwrite(file_path, pred_cell_pbc)


def pipeline_nuclei_matching(fname, persi_cell_boundary, persi_cell_list,
                             radius_cell, temp_dir, iou_th_nuclei_check):
    pred_nuclei_cellpose = imread(temp_dir + fname + '_nuclei_mask_3d.tiff')
    print('checking nuclei...')
    for p in persi_cell_list:
        pred_cell_pbc = imread(temp_dir + fname + '_pbc_' + p + '.tiff')
        idx_pred, idx_split_dic = [], {}
        temp = np.zeros(pred_nuclei_cellpose.shape, dtype=np.int32)
        temp[np.where(pred_nuclei_cellpose >= 1)] = 1
        idx_cell_list = np.unique(temp * pred_cell_pbc)[1:]
        cluster_merge_nuclei_dic = {}
        for idx_cell in idx_cell_list:
            # print(f'nuclei matching {idx_cell}')
            temp = np.zeros(pred_cell_pbc.shape, dtype=np.int32)
            temp[np.where(pred_cell_pbc == idx_cell)] = 1
            idx_nuclei_list = np.unique(temp * pred_nuclei_cellpose)[1:]
            is_cell = False
            nuclei_hit_list = []
            for idx_nuclei in idx_nuclei_list:
                if idx_nuclei not in cluster_merge_nuclei_dic:
                    cluster_merge_nuclei_dic[idx_nuclei] = []
                set_idx = set(zip(*np.where(pred_cell_pbc == idx_cell)))
                set_idx_nuclei = set(zip(*np.where(pred_nuclei_cellpose == idx_nuclei)))
                d_iou = intersection(set_idx_nuclei, set_idx)
                if d_iou > iou_th_nuclei_check:
                    cluster_merge_nuclei_dic[idx_nuclei].append(idx_cell)
                    # print(d_iou)
                    is_cell = True
                    nuclei_coor = np.where((pred_nuclei_cellpose == idx_nuclei) & (pred_cell_pbc == idx_cell))
                    nuclei_hit_list.append(nuclei_coor)
            if is_cell:
                idx_pred.append(idx_cell)
                if len(nuclei_hit_list) > 1:
                    idx_split_dic[idx_cell] = nuclei_hit_list
        print('splitting id')
        print(idx_split_dic.keys())
        # *** merge clusters overlapping the same nucleus***
        print(f'>>> merging clusters, p={p}')
        for idx in cluster_merge_nuclei_dic.keys():
            cell_list = cluster_merge_nuclei_dic[idx]
            print(cell_list)
            if len(cell_list) > 1:
                for i in range(1, len(cell_list)):
                    pred_cell_pbc[np.where(pred_cell_pbc == cell_list[i])] = cell_list[0]
                    if cell_list[i] in idx_split_dic:
                        v = idx_split_dic.pop(cell_list[i])
                        idx_split_dic[cell_list[0]] = v
        imwrite(temp_dir + fname + '_pbc_merge_' + p + '.tiff', pred_cell_pbc)

        '''Splitting clusters overlapping multiple nuclei'''
        pred_cell_low = imread(temp_dir + fname + '_pbc_' + persi_cell_boundary + '.tiff')
        pred_cell_pbc_nuclei = np.zeros(pred_cell_pbc.shape, dtype=np.int32)  # returning array of this step
        cell_id = 1

        for i in range(len(idx_pred)):
            if idx_pred[i] not in idx_split_dic:
                pred_cell_pbc_nuclei[np.where(pred_cell_pbc == idx_pred[i])] = cell_id
                cell_id += 1
            else:
                print(f'*** splitting cluster {idx_pred[i]}, p={p} ***')
                nuclei_hit_list = idx_split_dic[idx_pred[i]]
                pred_split = split_cluster_persi_boundary(fname, temp_dir, pred_cell_pbc, np.copy(pred_cell_low),
                                                          idx_pred[i], nuclei_hit_list, float(persi_cell_boundary))
                for j in range(1, int(np.max(pred_split)) + 1):
                    pred_cell_pbc_nuclei[np.where(pred_split == j)] = cell_id
                    cell_id += 1

        # imwrite('test_split_' + p + '.tiff', pred_cell_pbc_nuclei)
        # pred_cell_pbc_nuclei_list.append(pred_cell_pbc_nuclei)
        pred_cell_pbc_nuclei = rearrange_idx(pred_cell_pbc_nuclei)
        imwrite(temp_dir + fname + '_pbc_nuclei_' + p + '.tiff', pred_cell_pbc_nuclei)
        imwrite(temp_dir + fname + '_pbc_' + persi_cell_boundary + '.tiff', pred_cell_low)


def pipeline_output_w_pbc_cp(persi_cell_list, persi_cell_boundary, temp_dir, f_pred_dir, fname, padding):
    print('>>> saving predictions')

    if not os.path.exists(f_pred_dir):
        os.mkdir(f_pred_dir)

    pred_list = []
    for p in persi_cell_list[::-1]:
        pred_cell_pbc_nuclei = imread(temp_dir + fname + '_pbc_nuclei_' + p + '.tiff')
        pred_temp_list = []
        for i in range(1, int(np.max(pred_cell_pbc_nuclei) + 1)):
            pred_temp_list.append(np.where(pred_cell_pbc_nuclei == i))
        pred_list += pred_temp_list
    pred_list = arrange_by_persistence(pred_list)

    pred_cell_low = imread(temp_dir + fname + '_pbc_' + persi_cell_boundary + '.tiff')

    for i in range(len(pred_list)):

        pred_cell_low_copy = np.zeros(pred_cell_low.shape, dtype=np.uint16)
        pred_cell_low_copy[pred_list[i][0]] = pred_cell_low[pred_list[i][0]]
        pred_cell_cropped, x_min, x_max, y_min, y_max, z_min, z_max = crop_by_idx(pred_cell_low_copy, pred_list[i][0],
                                                                                  padding=padding)
        pred_cell_cropped = rearrange_idx(pred_cell_cropped)
        pred_cell_final = np.zeros([len(pred_list[i])] + list(pred_cell_cropped.shape), dtype=np.uint16)
        pred_cell_final[0] = pred_cell_cropped

        for j in range(1, len(pred_list[i])):
            pred_cell_low_copy = np.zeros(pred_cell_low.shape, dtype=np.uint16)
            pred_cell_low_copy[pred_list[i][j]] = pred_cell_low[pred_list[i][j]]
            pred_cell_final[j] = rearrange_idx(pred_cell_low_copy[z_min:z_max, x_min:x_max, y_min:y_max])

        dic_cell = {'idx_boundary': (x_min, x_max, y_min, y_max, z_min, z_max),
                    'idx': pred_list[i][0],
                    'pred': pred_cell_final,
                    'pbc': pred_cell_low[z_min:z_max, x_min:x_max, y_min:y_max]}

        with open(os.path.join(f_pred_dir, 'cell_' + str(i + 1) + '.pkl'), 'wb') as f:
            pickle.dump(dic_cell, f)


class Visualizer:

    def __init__(self, f_data, f_pred_dir, f_save, radius_cell, scale_dim=(1, 1, 1), data=None):

        self.cell_idx = 1
        self.last_cell_idx = self.cell_idx
        self.save_type = np.float32
        self.load_type_pred = np.int16
        self.history = {}
        self.cellGoToPanel = None
        self.layer_data_list = []
        self.layer_label_list = []
        self.has_existing_edits = False
        self.pred_pbc = None
        self.idx_mask_cell = None
        self.x_cropped = None
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = None, None, None, None, None, None
        self.f_pred_clusters = None
        self.f_cellname = None
        self.filename = None
        self.dic_cell = None
        self.pred_cropped = None

        self.f_data = f_data
        self.f_pred_dir = f_pred_dir
        self.f_save = f_save
        self.radius_cell = radius_cell
        self.scale_dim = scale_dim
        self.data = data
        self.f_save_final = f_save + f_data.split('/')[-1].replace('.tif', '_pred.tiff')
        self.f_save_final_separate = f_save + f_data.split('/')[-1].replace('.tif', '_pred_separate.tiff')
        self.f_save_final_mask = self.f_save_final.replace('.tiff', '_mask.tiff')
        self.x = imread(f_data)
        self.f_interactive_dir = f_pred_dir + '_interactive'

        if not os.path.exists(self.f_interactive_dir):
            os.mkdir(self.f_interactive_dir)

        if not os.path.exists(self.f_save_final):
            self.pred_final = np.zeros([self.x.shape[0], 3, self.x.shape[2], self.x.shape[3]])
            self.pred_final[:, :2, :, :] = np.copy(self.x)
            imwrite(self.f_save_final, self.pred_final)
            self.pred_final_mask = np.zeros([self.x.shape[0], self.x.shape[2], self.x.shape[3]],
                                            dtype=self.load_type_pred)
            imwrite(self.f_save_final_mask, self.pred_final_mask)
        else:
            self.pred_final = imread(self.f_save_final)
            self.pred_final_mask = imread(self.f_save_final_mask).astype(self.load_type_pred)

        self.num_cells = len(os.listdir(f_pred_dir))
        self.viewer = None

    def visualize(self):

        if not self.num_cells:
            print('No Cells Predicted')
            return

        while 1:

            if self.cellGoToPanel is not None and self.cellGoToPanel.jump:
                self.cell_idx = self.cellGoToPanel.idx
            if self.cell_idx == -1:
                self.viewer = napari.Viewer(title='All cells', ndisplay=3)
                # viewer.title = 'All cells'
                layer_show_all = self.viewer.add_image(self.pred_final,
                                                       channel_axis=1,
                                                       name=['nuclei', 'microglia', 'predictions'],
                                                       colormap=['blue', 'green', 'red'],
                                                       scale=self.scale_dim)
                layer_pred = layer_show_all[-1]

                @layer_pred.mouse_move_callbacks.append
                def mouse_move_callback(layer_pred, event):
                    start_point, end_point = layer_pred.get_ray_intersections(
                        position=event.position,
                        view_direction=event.view_direction,
                        dims_displayed=event.dims_displayed,
                        world=True
                    )
                    if start_point is not None and end_point is not None:
                        cur_cell_idx = None
                        u = end_point - start_point
                        u_max = np.max(np.abs(u))
                        u = u / u_max
                        # pred_cur = layer_label.data
                        for j in np.arange(1, u_max - 2, 2):
                            coor = (start_point + j * u).astype(np.uint16)
                            temp = self.pred_final[coor[0], 2, coor[1], coor[2]]
                            if temp != 0:
                                cur_cell_idx = self.pred_final_mask[coor[0], coor[1], coor[2]]
                                break
                        self.viewer.status = f'Coordinates: {coor}, Val:{temp}, Cell id: {cur_cell_idx}'

                for layer in layer_show_all:
                    @layer.bind_key('b', overwrite=True)
                    def show_all_back(layer):
                        self.cell_idx = self.last_cell_idx
                        self.layer_label_list = []
                        self.viewer.close()

                    @layer.bind_key('Escape', overwrite=True)
                    def show_all_terminate(layer):
                        self.viewer.close()
                        sys.exit(0)

            else:
                self.f_cell_name = f'cell_{self.cell_idx}.pkl'
                self.file_name = os.path.join(self.f_pred_dir, self.f_cell_name)
                # print(file_name)
                self.f_pred_clusters = os.path.join(self.f_interactive_dir, self.f_cell_name + '_pred_clusters.tiff')
                with open(self.file_name, 'rb') as f:
                    self.dic_cell = pickle.load(f)
                    self.pred_cropped = self.dic_cell['pred']
                    self.pred_pbc = self.dic_cell['pbc']
                    self.pred_cropped = [self.pred_cropped[0]]
                    self.idx_mask_cell = self.dic_cell['idx']
                    self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.dic_cell[
                        'idx_boundary']

                self.x_cropped = self.x[self.z_min:self.z_max, :, self.x_min:self.x_max, self.y_min:self.y_max]
                self.viewer = napari.Viewer(title=f'{self.f_pred_dir}_cell_{self.cell_idx}/{self.num_cells}',
                                            ndisplay=3)

                # viewer.title = f'{f_pred_dir}_cell_{cell_idx}/{num_cells}'

                instructPanel = CellEditingPanel()
                self.viewer.window.add_dock_widget(instructPanel, name='Interactive Commands',
                                                   add_vertical_stretch=True)

                self.layer_data_list = self.viewer.add_image(self.x_cropped,
                                                             channel_axis=1,
                                                             name=['nuclei', 'microglia'],
                                                             colormap=['blue', 'red'],
                                                             blending='additive',
                                                             scale=self.scale_dim)

                for persi_idx in range(len(self.pred_cropped)):
                    layer_label = self.viewer.add_labels(np.copy(self.pred_cropped[persi_idx]), name='pred_original')
                    layer_label.scale = self.scale_dim
                    self.layer_label_list.append(layer_label)
                    self.history[layer_label.name] = []

                if os.path.exists(self.f_pred_clusters):
                    pred_saved = imread(self.f_pred_clusters)
                else:
                    pred_saved = self.pred_cropped[0]

                x_cell_masked = np.copy(self.x_cropped[:, 1, :, :])
                x_cell_masked[np.where(pred_saved == 0)] = 0
                layer_cell_masked = self.viewer.add_image(x_cell_masked, name='microglia_edit',
                                                          colormap='green',
                                                          scale=self.scale_dim,
                                                          blending='additive')
                self.layer_label_list.append(layer_cell_masked)

                layer_label = self.viewer.add_labels(np.copy(pred_saved), name='pred_edit')
                layer_label.scale = self.scale_dim
                self.layer_label_list.append(layer_label)
                self.history[layer_label.name] = []
                self.layer_label_list[0].visible = False

                cellGoToPanel = CellEditingGoTo(str(self.cell_idx), self.num_cells, self.viewer, self)
                self.viewer.window.add_dock_widget(cellGoToPanel, name='Cell Navigator', add_vertical_stretch=True)
                cellSavePanel = CellEditingSave(self)
                self.viewer.window.add_dock_widget(cellSavePanel, name='Saving Options', add_vertical_stretch=True)
                for pred_layer_idx in range(len(self.layer_label_list)):
                    layer_label = self.layer_label_list[pred_layer_idx]

                    if layer_label.name == 'pred_original' or layer_label.name == 'pred_edit':

                        @layer_label.mouse_drag_callbacks.append
                        def on_right_click(layer_label, event):
                            if event.button == 2:
                                start_point, end_point = layer_label.get_ray_intersections(
                                    position=event.position,
                                    view_direction=event.view_direction,
                                    dims_displayed=event.dims_displayed,
                                    world=True
                                )
                                cluster_id = None
                                u = end_point - start_point
                                u_max = np.max(np.abs(u))
                                u = u / u_max
                                pred_cur = layer_label.data
                                for j in np.arange(1, u_max - 2, 2):
                                    coor = (start_point + j * u).astype(np.uint16)
                                    temp = pred_cur[coor[0], coor[1], coor[2]]
                                    if temp != 0:
                                        cluster_id = temp
                                        break

                                if cluster_id is not None:

                                    def pbc_mid_tau_worker():
                                        x_idx = np.where(pred_cur == cluster_id)
                                        pred_split = pbc_mid_tau(self.x_cropped[:, 1, :, :],
                                                                 x_idx)
                                        # print(np.max(pred_split))
                                        pred_cur[x_idx] = pred_split + np.max(pred_cur)
                                        layer_label.data = pred_cur
                                        self.pred_pbc[x_idx] = pred_split + np.max(self.pred_pbc)
                                        # print(np.unique(pred_cur))
                                        # print(np.unique(pred_pbc))
                                        self.history[layer_label.name].append(('split', cluster_id, x_idx))

                                    worker = create_worker(pbc_mid_tau_worker)
                                    worker.start()
                                    notif.show_info(f'Splitting cluster {cluster_id} ...')

                                else:
                                    notif.show_info('No cluster selected!')

                        @layer_label.mouse_double_click_callbacks.append
                        def on_second_click_of_double_click(layer_label, event):
                            start_point, end_point = layer_label.get_ray_intersections(
                                position=event.position,
                                view_direction=event.view_direction,
                                dims_displayed=event.dims_displayed,
                                world=True
                            )
                            pred_cur = layer_label.data  # current prediction mask
                            cluster_coor = None
                            if len(event.dims_displayed) == 3:
                                u = end_point - start_point
                                u_max = np.max(np.abs(u))
                                u = u / u_max
                                for j in np.arange(1, u_max - 2, 2):
                                    coor = (start_point + j * u).astype(np.uint16)
                                    temp = self.pred_pbc[coor[0], coor[1], coor[2]]
                                    if temp != 0:
                                        cluster_coor = coor
                                        break
                            else:
                                zz, xx, yy = round(event.position[0] / self.scale_dim[0]), round(
                                    event.position[1]), round(event.position[2])
                                if 0 <= xx <= pred_cur.shape[1] and 0 <= yy <= pred_cur.shape[2]:
                                    cluster_coor = (zz, xx, yy)
                            if cluster_coor is not None:
                                if pred_cur[cluster_coor[0], cluster_coor[1], cluster_coor[2]] != 0:
                                    cluster_id = pred_cur[cluster_coor[0], cluster_coor[1], cluster_coor[2]]
                                    notif.show_info(f'Cluster {cluster_id} removed!')
                                    idx = np.where(pred_cur == cluster_id)
                                    pred_cur[idx] = 0
                                    self.history[layer_label.name].append(('remove', cluster_id, idx))
                                    layer_label.data = pred_cur
                                else:
                                    val_pred_final_mask = self.pred_final_mask[cluster_coor[0] + self.z_min,
                                                                               cluster_coor[1] + self.x_min,
                                                                               cluster_coor[2] + self.y_min]
                                    if val_pred_final_mask != 0 and val_pred_final_mask != self.cell_idx:
                                        notif.show_error(f'Cluster is already assigned to Cell {val_pred_final_mask}')
                                    else:
                                        idx = np.where(
                                            self.pred_pbc == self.pred_pbc[
                                                cluster_coor[0], cluster_coor[1], cluster_coor[2]])
                                        cluster_id = np.max(pred_cur) + 1
                                        pred_cur[idx] = cluster_id
                                        layer_label.data = pred_cur
                                        self.history[layer_label.name].append(('add', 0, idx))
                                        notif.show_info(f'Cluster {cluster_id} added!')
                            else:
                                notif.show_info('No cluster selected!')

                        @layer_label.bind_key(key='u', overwrite=True)
                        def undo_remove(layer_label):
                            def undo_worker():
                                if len(self.history[layer_label.name]) == 0:
                                    return
                                action_type, cluster_id, idx = self.history[layer_label.name].pop()
                                pred_cur = layer_label.data
                                pred_cur[idx] = cluster_id
                                layer_label.data = pred_cur
                                if action_type == 'split':
                                    self.pred_pbc[idx] = cluster_id

                            worker = create_worker(undo_worker)
                            worker.start()
                            notif.show_info(f'Undo action')

                        @layer_label.bind_key(key='s', overwrite=True)
                        def save_pred(layer_label):
                            # notif.show_info(f'Predictions saved to {f_save_final}')
                            # notif.show_info(f'Saving predictions')

                            self.last_cell_idx = self.cell_idx
                            if not os.path.exists(self.f_save):
                                os.makedirs(self.f_save)

                            pred_cur = layer_label.data

                            ids_overlap = self.check_duplicate(pred_cur)

                            if len(ids_overlap) > 0:
                                # notif.ErrorNotification(f'{ids_overlap} assigned to existing cells')
                                error_msg = ''
                                for id in ids_overlap:
                                    idx = np.where(pred_cur == id)
                                    other_id = self.pred_final_mask[self.z_min:self.z_max, self.x_min:self.x_max,
                                               self.y_min:self.y_max][idx[0][0], idx[1][0], idx[2][0]]
                                    error_msg += f'Cluster {id} is already assigned to Cell {other_id}; '
                                notif.show_error(error_msg)

                            else:
                                notif.show_info(f'Edits Saved')
                                imwrite(self.f_pred_clusters, pred_cur)  # saved edited clusters

                                idx_pos = np.where(pred_cur > 0)
                                # pred_cur[idx_pos] = 1
                                new_idx_mask = list(np.copy(idx_pos))
                                new_idx_mask[0] += self.z_min
                                new_idx_mask[1] += self.x_min
                                new_idx_mask[2] += self.y_min
                                new_idx_mask = tuple(new_idx_mask)

                                pred_cell_channel = self.pred_final[:, 2, :, :]
                                pred_cell_channel[self.idx_mask_cell] = 0
                                self.pred_final_mask[self.idx_mask_cell] = 0

                                pred_cell_channel[new_idx_mask] = self.x_cropped[:, 1, :, :][
                                    idx_pos]  # save the pred. in original intensity
                                self.pred_final_mask[new_idx_mask] = self.cell_idx

                                self.dic_cell['idx'] = new_idx_mask
                                self.dic_cell['pbc'] = self.pred_pbc
                                with open(self.file_name, 'wb') as f:
                                    pickle.dump(self.dic_cell, f)
                                self.reload_cell(reset_view=False)

                    @layer_label.bind_key(key='Right', overwrite=True)
                    def next_cell(layer_label):
                        self.last_cell_idx = self.cell_idx
                        if self.cell_idx < self.num_cells:
                            self.cell_idx += 1
                        else:
                            self.cell_idx = 1
                        self.reload_cell()
                        cellGoToPanel.lineEdit.setText(str(self.cell_idx))

                    @layer_label.bind_key(key='Left', overwrite=True)
                    def prev_cell(layer_label):
                        self.last_cell_idx = self.cell_idx
                        if self.cell_idx > 1:
                            self.cell_idx -= 1
                        else:
                            self.cell_idx = self.num_cells
                        self.reload_cell()
                        cellGoToPanel.lineEdit.setText(str(self.cell_idx))

                    @layer_label.bind_key(key='a', overwrite=True)
                    def show_all(layer_label):
                        self.last_cell_idx = self.cell_idx
                        worker = create_worker(self.save_as_one)
                        worker.start()
                        self.viewer.close()
                        self.cell_idx = -1

                    @layer_label.bind_key(key='Escape', overwrite=True)
                    def terminate(layer_label):
                        print('saving ...')
                        self.viewer.close()
                        self.save_as_one()
                        # self.save_separate()
                        sys.exit(0)
            napari.run()

    def reload_cell(self, reset_view=True):
        # self.history = {}
        for key in self.history:
            self.history[key] = []
        self.f_cell_name = f'cell_{self.cell_idx}.pkl'
        self.file_name = os.path.join(self.f_pred_dir, self.f_cell_name)
        self.f_pred_clusters = os.path.join(self.f_interactive_dir, self.f_cell_name + '_pred_clusters.tiff')
        with open(self.file_name, 'rb') as f:
            self.dic_cell = pickle.load(f)
            self.pred_cropped = self.dic_cell['pred']
            self.pred_pbc = self.dic_cell['pbc']
            self.pred_cropped = [self.pred_cropped[0]]
            self.idx_mask_cell = self.dic_cell['idx']
            self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.dic_cell['idx_boundary']

        self.x_cropped = self.x[self.z_min:self.z_max, :, self.x_min:self.x_max, self.y_min:self.y_max]
        # pred_mask_cropped = self.pred_final_mask[self.z_min:self.z_max, self.x_min:self.x_max, self.y_min:self.y_max]
        # self.viewer.add_labels(pred_mask_cropped)
        self.layer_data_list[0].data = self.x_cropped[:, 0, :, :]
        self.layer_data_list[1].data = self.x_cropped[:, 1, :, :]

        for persi_idx in range(len(self.pred_cropped)):
            self.layer_label_list[persi_idx].data = np.copy(self.pred_cropped[persi_idx])
            # self.history[self.layer_label_list[persi_idx].name] = []

        if os.path.exists(self.f_pred_clusters):
            pred_saved = imread(self.f_pred_clusters)
        else:
            pred_saved = self.pred_cropped[0]

        # pred_saved = self.check_duplicate(pred_saved, pred_mask_cropped)

        x_cell_masked = np.copy(self.x_cropped[:, 1, :, :])
        x_cell_masked[np.where(pred_saved == 0)] = 0

        self.layer_label_list[1].data = x_cell_masked
        self.layer_label_list[2].data = pred_saved

        # self.layer_label_list[3].data = self.pred_pbc ### debugging purpose

        # self.history[self.layer_label_list[-1].name] = []
        self.viewer.title = f'{self.f_pred_dir}_cell_{self.cell_idx}/{self.num_cells}'

        if reset_view:
            self.viewer.reset_view()

    def check_duplicate(self, pred):
        pred_mask_cropped = np.copy(
            self.pred_final_mask[self.z_min:self.z_max, self.x_min:self.x_max, self.y_min:self.y_max])
        pred_mask_cropped[np.where(pred_mask_cropped == self.cell_idx)] = 0
        pred_mask_cropped[np.where(pred_mask_cropped > 0)] = 1
        ids_overlap = np.unique(pred_mask_cropped * pred)[1:]
        return ids_overlap

    def save_as_one(self):
        imwrite(self.f_save_final, self.pred_final.astype(self.save_type))
        imwrite(self.f_save_final_mask, self.pred_final_mask.astype(self.save_type))

    def save_separate(self):
        res = []
        # pred_separate = np.zeros((self.num_cells, self.x.shape[0], self.x.shape[2], self.x.shape[3]), dtype=np.int16)
        # for i in range(self.num_cells):
        for i in range(self.num_cells):
            idx = np.where(self.pred_final_mask == i + 1)
            if len(idx[0]) != 0:
                pred_cell = np.zeros_like(self.x[:, 1, :, :])
                pred_cell[idx] = self.x[:, 1, :, :][idx]
                res.append(pred_cell)

        res = np.transpose(np.array(res), [1, 0, 2, 3])
        imwrite(self.f_save_final_separate, np.concatenate([self.x, res], axis=1))


def remove_old_pred(f_data, f_save, f_pred_dir):
    if os.path.exists(f_pred_dir):
        flist = os.listdir(f_pred_dir)
        for f in flist:
            os.remove(os.path.join(f_pred_dir, f))

    if os.path.exists(f_pred_dir + '_interactive'):
        flist = os.listdir(f_pred_dir + '_interactive')
        for f in flist:
            os.remove(os.path.join(f_pred_dir + '_interactive', f))

    f_save_final = f_save + f_data.split('/')[-1].replace('.tif', '_pred.tiff')
    if os.path.exists(f_save_final):
        os.remove(f_save_final)


def save_to_file(file_path, visualization_only, nuclei_ch, otsu_cell, otsu_nuclei, pixel_x, pixel_z, cellpose_diameter,
                 persi_str, vis_pad_size, z_len, f_config, dic_config):
    lines = [
             f'{visualization_only}',
             f'{nuclei_ch}',
             f'{otsu_cell} {otsu_nuclei}',
             f'{persi_str}',
             f'{pixel_x} {pixel_z}',
             f'{vis_pad_size}',
             f'{z_len}',
             f'{cellpose_diameter}',
             ]

    with open(f_config, 'wb') as f:
        dic_config[file_path] = lines
        dic_config['@last_file'] = file_path
        print(dic_config)
        pickle.dump(dic_config, f)


def launch_func(interactive_dir, cellpose_dir, save_dir, temp_dir, f_config='gui_config.pkl'):
    last_config = None
    default_path = ''
    if os.path.exists(f_config):
        with open(f_config, 'rb') as f:
            dic_config = pickle.load(f)
            if '@last_file' in dic_config:
                default_path = dic_config['@last_file']
                last_config = dic_config[default_path]
    else:
        with open(f_config, 'wb') as f:
            dic_config = {}
            pickle.dump(dic_config, f)

        # with open('gui_config.txt') as f:
        #     last_config = [line[:-1] for line in f]
    # default_path = ''
    # if last_config is not None:
    #     default_path = last_config[0]
    app = QApplication(sys.argv)
    menu = FileSelector(default_path, save_dir, last_config, idx=0)
    menu.show()
    app.exec_()
    f_data = menu.getPaths()
    data = imread(f_data)
    file_path = menu.getPaths()
    visualization_only = menu.getModeIndex()
    if file_path in dic_config:
        last_config = dic_config[file_path]
        if len(last_config) != 8:
            last_config = None
    else:
        last_config = None

    menu = Launch(tuple(data.shape), config=last_config, idx=1)
    if not visualization_only:
        menu.show()
        app.exec_()
    nuclei_ch = menu.nucleiCH.getIndex()
    otsu_cell, otsu_nuclei = menu.otsu.getValue()
    pixel_x, pixel_z = menu.pixelSize.getValue()
    cellpose_diameter = menu.diameter.getValue()
    persi_str = menu.perciCell.getValue()
    vis_pad_size = menu.paddingSize.getValue()
    z_len = menu.zBatch.getValue()
    persi_str = persi_str.replace(' ', '')
    save_to_file(file_path, visualization_only, nuclei_ch, otsu_cell, otsu_nuclei, pixel_x, pixel_z, cellpose_diameter,
                 persi_str, vis_pad_size, z_len, f_config, dic_config)
    spl = persi_str.split(';')
    persi_cell_list = [e for e in spl if e != '']
    # persi_cell_list = list(map(float, persi_str.split(';')))
    scale_dim = [pixel_z / pixel_x, 1, 1]

    fname = f_data.split('/')[-1]

    print('processing %s' % str(fname))
    print('Input dim. %s' % str(data.shape))

    if not os.path.exists(interactive_dir):
        os.mkdir(interactive_dir)

    if not os.path.exists(cellpose_dir):
        os.mkdir(cellpose_dir)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    shape = list([data.shape[0]]) + list(data.shape[2:])
    f_pred_dir = os.path.join(save_dir, fname)

    if nuclei_ch == 1:
        nuclei = np.expand_dims(data[:, 1, :, :], axis=1)
        cell = np.expand_dims(data[:, 0, :, :], axis=1)
        data = np.concatenate((nuclei, cell), axis=1)

    return data, f_data, fname, f_pred_dir, shape, nuclei_ch, otsu_cell, otsu_nuclei, scale_dim, vis_pad_size, persi_cell_list, z_len, cellpose_diameter, visualization_only


def remove_temp_files(temp_dir):
    flist = os.listdir(temp_dir)
    for f in flist:
        print(os.path.join(temp_dir, f))
        os.remove(os.path.join(temp_dir, f))


def inference(iou_th_nuclei_check=0.01, radius_cell='1.8', save_dir='predictions_v3/', temp_dir='temp/',
              interactive_dir='interactive/', cellpose_dir='cellpose/', save_intermediate=True):
    data, f_data, fname, f_pred_dir, shape, nuclei_ch, otsu_cell, otsu_nuclei, scale_dim, padding, persi_list, z_len, cellpose_diameter, visualize_only = launch_func(
        interactive_dir, cellpose_dir, save_dir, temp_dir)

    print(f'nuclei ch {nuclei_ch}')
    print(f'otsu_cell {otsu_cell}')
    print(f'scale_dim {scale_dim}')
    print(f'padding {padding}')
    print(f'persi_list {persi_list}')
    print(f'z_len {z_len}')
    print(f'cellpose_diameter {cellpose_diameter}')
    print(f'visualize_only {visualize_only}')

    if nuclei_ch == 1:
        nuclei = data[:, 1, :, :]
        cell = data[:, 0, :, :]
        data = np.concatenate((np.expand_dims(nuclei, 1), np.expand_dims(cell, 1)), axis=1)

    persi_list.sort()  # perse_cell_list must be in increasing order

    print(persi_list)

    persi_cell_boundary = persi_list[0]
    persi_cell_list = persi_list

    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # data, fname, f_pred_dir, shape = pipeline_setup(f_data, interactive_dir, cellpose_dir, save_dir)

    if z_len > len(data):
        z_len = len(data)

    if not visualize_only:
        pipeline_preprocessing(fname, data, interactive_dir, cellpose_dir, temp_dir, otsu_cell,
                               otsu_nuclei, cellpose_diameter, z_len)

        pipeline_cell_pbc_gudhi(fname, persi_list, radius_cell, temp_dir)

        pipeline_nuclei_matching(fname, persi_cell_boundary, persi_cell_list,
                                 radius_cell, temp_dir, iou_th_nuclei_check)

        remove_old_pred(f_data, save_dir, f_pred_dir)
        pipeline_output_w_pbc_cp(persi_cell_list, persi_cell_boundary, temp_dir, f_pred_dir, fname, padding)

        if not save_intermediate:
            remove_temp_files(temp_dir)

    visualizer = Visualizer(f_data=f_data, f_pred_dir=f_pred_dir, f_save=save_dir, radius_cell=radius_cell,
                            scale_dim=scale_dim)
    visualizer.visualize()

    print('time elapsed: %.2f sec' % (time.time() - start_time))


if __name__ == '__main__':
    inference(save_dir='predictions_test/', temp_dir='temp/', save_intermediate=False)

