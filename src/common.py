import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def map_sequence(input_list):
    num_to_order = {}
    order = 0
    result = []
    for num in input_list:
        if num not in num_to_order:
            num_to_order[num] = order
            order += 1
        result.append(num_to_order[num])
    return result

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def compute_target_prototypes(H_lst, new_target, num_classes):
    # H_lst: List[List[tensor]], new_target: List[int]
    prototypes = [] # 原型向量
    vectors = []
    labels = []
    for c in range(num_classes):
        prototype = []
        for i in range(len(new_target)):
            if new_target[i] == c:
                prototype.extend([utt for utt in H_lst[i]]) # 取所有子对话的向量
                vectors.extend([utt for utt in H_lst[i]]) # 取所有子对话的向量，一视同仁地放在列表里，flatten展开
                labels.extend([c for utt in H_lst[i]])
        if prototype:
            prototype = torch.stack(prototype).mean(dim=0) # 取当前对话的所有子对话的平均向量
        else:
            prototype = torch.zeros_like(H_lst[0][0])
        prototypes.append(prototype)
    return torch.stack(prototypes), torch.stack(vectors), torch.tensor(labels)

def target_CL(H_lst, target, config):
    # H_lst: List of utterance list; target: list of int/string
    tar_dic = {} # 目标的字典投影
    tmp = 0
    for t in target:
        if t not in tar_dic:
            tar_dic[t] = tmp
            tmp += 1
    new_target = [tar_dic[t] for t in target] # 长度为目标数量的等差数列
    num_classes = tmp
    prototypes, vectors, labels = compute_target_prototypes(H_lst, new_target, num_classes) # prototypes: (B, D) 每个对话中所有子对话的平均向量; vectors: (所有单对话,D)，只是取向量
    vectors = F.normalize(vectors, p=2, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    similarities = torch.mm(vectors, prototypes.T) # (所有单对话,D)*(D, B) = (所有单对话, 4)
    similarities /= config.tau # 每一行是某个子对话向量与所有类别原型的余弦相似度
    labels = labels.to(config.device)
    loss = F.cross_entropy(similarities, labels) # 基于类别原型的对比学习分类 >——> 每个子对话向量被要求与自己类别的原型向量最接近，同时和其他类别的原型保持距离
    return loss
