import os
import numpy as np
import gzip
import json
import datetime
import pdb
from datetime import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import pickle
import sys
import logging
from logging import *
import shutil
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
from collections import OrderedDict
import random
from tqdm import tqdm


_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(
    level=logging.INFO, format=_FMT, datefmt=_DATEFMT, stream=sys.stdout
)


def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

  
def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if path is not None:
        from pathlib import Path
        path = str(Path(path).resolve())
        if not any(
            map(lambda hdr: hasattr(hdr, 'baseName') 
                and hdr.baseFilename == path, logger.handlers)):
            handler = fileHandler(path, format, datefmt)
            logger.addHandler(handler)

    return logger


def save_checkpoint(state, is_best, dir='checkpoints/', name='checkpoint'):
    os.makedirs(dir, exist_ok=True)
    filename = dir + name + f"_e{state['epoch']}" + '.pth'
    torch.save(state, filename)
    if is_best:
        best_filename = dir + name + '_model_best.pth'
        shutil.copyfile(filename, best_filename)
        
        
def save_checkpoint_best_only(state, dir='checkpoints/', name='checkpoint'):
    os.makedirs(dir, exist_ok=True)
    best_filename = os.path.join(dir, name + '_model_best.pth')
    torch.save(state, best_filename)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)

    return images

    
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

 
def trim(stat, prefix='module'):
    r"""Remove prefix of state_dict keys.
    """

    stat_new = OrderedDict()
    for k, v in stat.items():
        if k.startswith(prefix):
            stat_new[k[len(prefix)+1:]] = v

    return stat_new if stat_new else stat


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def plot_confusion_matrix(cm, classes, plot_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, logger=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if logger:
            logger.info("Normalized confusion matrix")
        else:
            print("Normalized confusion matrix")
    else:
        if logger:
            logger.info('Confusion matrix, without normalization')
        else:
            print('Confusion matrix, without normalization')

    if logger:
        logger.info(cm)
    else:
        print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()
    
    if logger:
        logger.info('{} saved!'.format(plot_name))
    else:
        print('{} saved!'.format(plot_name))    
    return


def write_list_of_strings_into_a_txtfile(lines, savepath):
    textfile = open(savepath, "w")
    for line in lines:
        textfile.write(line + "\n")
    textfile.close()
    return


def collect_results_for_analysis(results, class2idx, 
                                 dataset_name='volleyball', 
                                 result_prefix='./result',
                                 person_action_predicted=True,
                                 logger=None):
    idx2class = dict()
    if dataset_name == 'volleyball':
        for c in class2idx:
            if '_' in c:
                idx2class[class2idx[c]] = c
                if logger:
                    logger.info('{}: {}'.format(class2idx[c], c))
                  
        if person_action_predicted:
            person_idx2class = {1: 'blocking', 
               2: 'digging', 
               3: 'falling', 
               4: 'jumping',
               5: 'moving', 
               6: 'setting', 
               7: 'spiking', 
               8: 'standing',
               9: 'waiting',
               0: 'N/A'}

        else:
            person_idx2class = None
    else:
        for c in class2idx:
            idx2class[class2idx[c]] = c
            if logger:
                logger.info('{}: {}'.format(class2idx[c], c))
                
        person_idx2class = None
     
    if logger:
        logger.info('collecting results to files...')
        
    gt_all = []
    pred_all = []
    if person_idx2class:
        gt_person_all = []
        pred_person_all = []
    lines = []
    for key in tqdm(results):
        [videoid, clipid] = key.split('-')
        gt = idx2class[results[key]['GT']]
        pred = idx2class[results[key]['Pred']]
        gt_all.append(gt)
        pred_all.append(pred)
        
        if person_idx2class:
            results[key]['GT_Person'] = [person_idx2class[val] for val in results[key]['GT_Person']]
            results[key]['Pred_Person'] = [person_idx2class[val] for val in results[key]['Pred_Person']]
            gt_person_all += results[key]['GT_Person']
            pred_person_all += results[key]['Pred_Person']
            
        lines.append("{}  GT {} Pred {} Correct? {}".format(key, gt, pred, gt==pred))
    
    # # write results to disk
    pickle.dump(results, open(result_prefix+'.pkl', "wb"))
    
    # write results to txt file
    write_list_of_strings_into_a_txtfile(lines, result_prefix+'.txt')
    
    # get confusion matrix
    cnf_matrix = confusion_matrix(gt_all, pred_all, labels=list(idx2class.values()))
    np.set_printoptions(precision=2)

    # plot normalized confusion matrix
    plot_name = result_prefix + '.png'
    plot_confusion_matrix(cnf_matrix, list(idx2class.values()), plot_name, normalize=True,
                          title='Confusion matrix, with normalization', cmap=plt.cm.Reds, logger=logger)
    
    if person_idx2class:
        # plot normalized confusion matrix for person action prediction
        cnf_matrix_person = confusion_matrix(gt_person_all, pred_person_all, labels=list(person_idx2class.values()))
        np.set_printoptions(precision=2)

        # plot normalized confusion matrix for person action prediction
        plot_name = result_prefix + '_person.png'
        plot_confusion_matrix(cnf_matrix_person, list(person_idx2class.values()), plot_name, normalize=True,
                              title='Confusion matrix, with normalization', cmap=plt.cm.Reds, logger=logger)   

    return 
    
    
    
                
    
    
    