import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import random
import csv
import sys
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def debug(data, manager_p, manager, args):

    print('-----------------Data--------------------')
    data_attrs = ["data_dir","n_known_cls","num_labels","all_label_list","known_label_list"]

    for attr in data_attrs:
        attr_name = attr
        attr_value = data.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Args--------------------')
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])

    print('-----------------Manager_pretrain--------------------')
    manager_p_attrs = ["device","num_train_optimization_steps","best_eval_score"]

    for attr in manager_p_attrs:
        attr_name = attr
        attr_value = manager_p.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Manager--------------------')
    manager_attrs = ["device","best_eval_score","test_results"]

    for attr in manager_attrs:
        attr_name = attr
        attr_value = manager.__getattribute__(attr)
        print(attr_name,':',attr_value)
    
    if manager.predictions is not None:
        print('-----------------Predictions--------------------')
        show_num = 100
        for i,example in enumerate(data.test_examples):
            if i >= show_num:
                break
            sentence = example.text_a
            true_label = manager.true_labels[i]
            predict_label = manager.predictions[i]
            print(i,':',sentence)
            print('Pred:{}; True:{}'.format(predict_label,true_label))

def target2onehot(targets, num_classes):
    return torch.eye(num_classes)[targets]

def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Known'] = f_seen
    result['Open'] = f_unseen
    result['F1-score'] = f
        
    return result

def plot_confusion_matrix(cm, classes, save_name, normalize=False, title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues, save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.switch_backend('agg')
    
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        plt.savefig(save_name)
    
# def draw(x, y):
#     from matplotlib.colors import ListedColormap
#     from MulticoreTSNE import MulticoreTSNE as TSNE
#
#     print("TSNE: fitting start...")
#     tsne = TSNE(2, n_jobs=4, perplexity=30)
#     Y = tsne.fit_transform(x)
#
#     # matplotlib_axes_logger.setLevel('ERROR')
#     labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','open']
#     id_to_label = {i: label for i, label in enumerate(labels) }
#     y_true = pd.Series(y)
#     plt.style.use('ggplot')
#     n_class = y_true.unique().shape[0]
#     colors = ( 'gray','lightgreen', 'plum','DarkMagenta','SkyBlue','PaleTurquoise','DeepPink','Gold','Orange','Brown','DarkKhaki')
#
#     fig, ax = plt.subplots(figsize=(9, 6), )
#     la = [i for i in range(n_class)]
#     la = sorted(la,reverse=True)
#     cmap = ListedColormap(colors)
#     for idx, label in enumerate(la):
#         ix = y_true[y_true==label].index
#         x = Y[:, 0][ix]
#         y = Y[:, 1][ix]
#         ax.scatter(x, y, c=cmap(idx), label=id_to_label[label], alpha=0.5)
#
#     # Shrink current axis by 20%
#     ax.set_title('proto_loss')
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# def draw(x, y):
#     from matplotlib.colors import ListedColormap
#     from openTSNE import TSNE
#
#     print("TSNE: fitting start...")
#     tsne = TSNE(n_components=2, n_jobs=4, perplexity=30)
#     Y = tsne.fit(x)
#
#     labels = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'open']
#     unique_labels = sorted(set(y))  # 获取 y 中的所有唯一标签
#     id_to_label = {i: labels[i] if i < len(labels) else f'class_{i}' for i in unique_labels}  # 动态生成标签映射
#     y_true = pd.Series(y)
#     plt.style.use('ggplot')
#     n_class = y_true.unique().shape[0]
#     colors = ('gray', 'lightgreen', 'plum', 'DarkMagenta', 'SkyBlue', 'PaleTurquoise', 'DeepPink', 'Gold', 'Orange', 'Brown', 'DarkKhaki')
#
#     fig, ax = plt.subplots(figsize=(9, 6))
#     cmap = ListedColormap(colors)
#
#     for idx, label in enumerate(unique_labels):
#         ix = y_true[y_true == label].index
#         x_plot = Y[:, 0][ix]
#         y_plot = Y[:, 1][ix]
#         ax.scatter(x_plot, y_plot, c=cmap(idx), label=id_to_label[label], alpha=0.5)
#
#     ax.set_title('proto_loss')
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()

# 可以绘制5+1
# def draw(x, y):
#     from matplotlib.colors import ListedColormap
#     from openTSNE import TSNE
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#
#     print("TSNE: fitting start...")
#     tsne = TSNE(n_components=2, n_jobs=4, perplexity=30)
#     Y = tsne.fit(x)
#
#     # 获取传入的标签，确保是唯一的类名，并生成与之对应的颜色
#     unique_labels = np.unique(y)
#     num_classes = len(unique_labels)
#
#     # 生成一个调色板（tab10有10种颜色，如果类别数大于10，将重复使用颜色）
#     cmap = plt.get_cmap('tab10')  # 使用 tab10 调色板，不再指定 num_classes
#     colors = [cmap(i % 10) for i in range(num_classes)]  # 为每个类生成颜色，确保颜色数量足够
#
#     # 创建标签与颜色的映射
#     label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
#
#     y_true = pd.Series(y)
#     plt.style.use('ggplot')
#
#     fig, ax = plt.subplots(figsize=(9, 6))
#
#     # 绘制每个类别的 t-SNE 图
#     for label in unique_labels:
#         ix = y_true[y_true == label].index
#         x_plot = Y[:, 0][ix]
#         y_plot = Y[:, 1][ix]
#         ax.scatter(x_plot, y_plot, color=colors[label_to_idx[label]], label=label, alpha=0.5)
#
#     ax.set_title('t-SNE visualization of selected classes')
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()

# 可以绘制全部的类别
def draw(x, y):
    from openTSNE import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import colorcet as cc  # 使用 colorcet 以获取更多颜色

    print("TSNE: fitting start...")
    tsne = TSNE(n_components=2, n_jobs=8, perplexity=10)
    Y = tsne.fit(x)

    # 获取传入的标签，确保是唯一的类名
    unique_labels = np.unique(y)

    # 使用 colorcet 的大调色板来为已知类设置颜色
    num_classes = len([label for label in unique_labels if label != 'open'])
    colors = cc.glasbey[:num_classes]  # 使用 colorcet 提供的 glasbey 调色板，适合分类问题
    color_map = {label: colors[i] for i, label in enumerate(unique_labels) if label != 'open'}

    # 为未知类（open）设置五角星的标记
    marker_styles = {'open': 'p'}

    y_true = pd.Series(y)
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(9, 6))

    # 绘制已知类的样本
    for label in unique_labels:
        if label == 'open':
            continue  # 最后绘制 open 类
        ix = y_true[y_true == label].index
        x_plot = Y[:, 0][ix]
        y_plot = Y[:, 1][ix]
        ax.scatter(x_plot, y_plot, color=color_map[label], label='', alpha=0.7)

    # 最后绘制未知类（open）的样本，使用五角星
    if 'open' in unique_labels:
        ix = y_true[y_true == 'open'].index
        x_plot = Y[:, 0][ix]
        y_plot = Y[:, 1][ix]
        ax.scatter(x_plot, y_plot, color='red', marker=marker_styles['open'], label='open', alpha=0.7)

    ax.set_title('t-SNE visualization of known and unknown classes')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # 只显示 open 类在图例中
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


# def draw(x, y):
#     from matplotlib.colors import ListedColormap
#     from openTSNE import TSNE
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#
#     print("TSNE: fitting start...")
#     tsne = TSNE(n_components=2, n_jobs=4, perplexity=30)
#     Y = tsne.fit(x)
#
#     # 获取传入的标签，确保是唯一的类名，并生成与之对应的颜色
#     unique_labels = np.unique(y)
#     num_classes = len(unique_labels)
#
#     # 使用更鲜明的调色板，比如 'tab20' 或 'Set3'
#     cmap = plt.get_cmap('tab20')  # 可以尝试 'Set3', 'tab20', 'Paired' 等调色板
#     colors = [cmap(i % 20) for i in range(num_classes)]  # 确保有足够的颜色
#
#     # 创建标签与颜色的映射
#     label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
#
#     y_true = pd.Series(y)
#     plt.style.use('ggplot')
#
#     fig, ax = plt.subplots(figsize=(9, 6))
#
#     # 先绘制已知类，再绘制 open 类
#     for label in unique_labels:
#         if label == 'open':
#             continue  # 最后绘制 open 类
#         ix = y_true[y_true == label].index
#         x_plot = Y[:, 0][ix]
#         y_plot = Y[:, 1][ix]
#         ax.scatter(x_plot, y_plot, color=colors[label_to_idx[label]], label=label, alpha=0.7)
#
#     # 最后绘制 open 类
#     if 'open' in unique_labels:
#         ix = y_true[y_true == 'open'].index
#         x_plot = Y[:, 0][ix]
#         y_plot = Y[:, 1][ix]
#         ax.scatter(x_plot, y_plot, color='black', label='open', alpha=0.7)  # 将 open 类标记为黑色
#
#     ax.set_title('t-SNE visualization of selected classes')
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#     # 调整图例顺序，使 open 类在最后
#     handles, labels = ax.get_legend_handles_labels()
#     if 'open' in labels:
#         open_idx = labels.index('open')
#         handles.append(handles.pop(open_idx))
#         labels.append(labels.pop(open_idx))
#     ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.show()




def plot_curve(points):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    plt.grid(alpha=0.4)
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', 'H', 'v', '>', 'h', 'H', 'v', '>', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = np.arange(-0.02, len(centers[0]) + 0.01).astype(dtype=np.str)
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
    
    plt.xlim(0, 20, 1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(r'Decision Boundary $\Delta$', fontsize=12)
    plt.legend()
    plt.title('50% Known Classes on StackOverflow')
    plt.show()
    plt.savefig('curve.pdf')
    

