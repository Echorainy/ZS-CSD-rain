import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from box import Box
from src.preprocess import DataProcessor
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score, accuracy_score
from src.common import set_seed
from src.model import SITCL

class Main:
    def __init__(self, args):
        config = Box(yaml.load(open('D:/零样本立场/ZS-CSD/src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        #打开配置文件 → 解析 YAML → 包装成 Box → 保存到 config
        for k, v in vars(args).items():
            #取出args中每个参数的键值，放进 k 和 v。
            #vars(args)返回键值 .items()历遍键值
            setattr(config, k, v) #setattr根据字符串形式的名字设置对象的属性，setattr(对象, 属性名, 属性值)
        self.config = config
        self.formatted_time = config.time
        self.log_dir = f"./result/{self.config.model_type}_{self.formatted_time}/log/" #日志目录
        self.pred_dir = f"./result/{self.config.model_type}_{self.formatted_time}/pred/" #预测结果目录
        self.save_dir = f"./result/{self.config.model_type}_{self.formatted_time}/save/" #模型保存目录
        self.pred_file = self.pred_dir + f'{self.config.seed}.csv'
        for d in [self.log_dir, self.pred_dir, self.save_dir]:
            os.makedirs(d, exist_ok=True) #创建目录
        logging.basicConfig( #配置日志
            filename=self.log_dir + f'{self.config.seed}.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        set_seed(self.config.seed)
        self.config.device = torch.device('cuda:{}'.format(self.config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.best_epoch = 0
        self.best_test_macro_f1 = 0.0

    def train_iter(self): #遍历训练数据加载器一轮
        self.model.train() #把模型切换为训练模式
        running_loss = 0.0
        for data in tqdm(self.trainLoader):
            loss, _, _ = self.model(**data)
            loss.backward() #计算并累积梯度
            nn.utils.clip_grad_norm_( #缩小梯度
            	self.model.parameters(), 
                self.config.max_grad_norm) #梯度上限
            self.optimizer.step() #根据梯度更新参数
            self.scheduler.step() #调整学习率
            self.model.zero_grad() #清除梯度
            running_loss += loss.item() #累加损失
        train_loss = running_loss / len(self.trainLoader)
        return train_loss

    def evaluate_iter(self, dataLoader=None, mode=None):
        self.model.eval()
        val_loss = 0.0
        seq_preds = []
        seq_trues = []
        doc_id_lst = []
        if mode == 'dev':
            dataLoader = self.devLoader
        elif mode == 'test':
            dataLoader = self.testLoader
        with torch.no_grad():
            for data in tqdm(dataLoader):
                loss, seq_output, labels = self.model(**data)
                val_loss += loss.item()
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                labels = labels.detach().cpu().numpy()
                labels = labels.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(labels)
                doc_id_lst.extend(data['doc_id'])
        val_loss /= len(dataLoader)
        macro_f1, favor, against, neutral, f1_avg, acc = self.get_metrices(seq_trues, seq_preds)
        if mode == 'test':
            result = {'doc_id': doc_id_lst, 'true': seq_trues, 'pred': seq_preds}
            df = pd.DataFrame(result)
            df.to_csv(self.pred_file, index=False)
        return macro_f1, val_loss, favor, against, neutral, f1_avg, acc

    def train(self):
        best_dev_f1 = 0.0 #最佳验证集 macro F1
        for epoch in range(self.config.epoch_size):
            train_loss = self.train_iter() #更新训练集损失参数
            dev_macro_f1, dev_loss, *_ = self.evaluate_iter(mode='dev') #验证集评估模型
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.2f}, Val Loss: {dev_loss:.2f}, Val Macro F1: {100 * dev_macro_f1:.2f}')
            if dev_macro_f1 >= best_dev_f1:
                best_dev_f1 = dev_macro_f1
                test_macro_f1, test_loss, *_ = self.evaluate_iter(mode='test')
                logging.info(f'Test Loss: {test_loss:.2f}, Test Macro F1: {100 * test_macro_f1:.2f}')
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_dir + 'best_model.pth')

    def load_param(self):
        param_optimizer = list(self.model.named_parameters()) #模型参数名、参数对象
        no_decay = ['bias', 'LayerNorm.weight'] #不给偏置、层归一化施加权重衰减
        bert_lr = float(self.config.bert_lr) #两组学习率
        other_lr = float(self.config.other_lr)
        optimizer_grouped_parameters = [ #分组设置权重衰减、学习率
            {'params': [p for n, p in param_optimizer if 'bert' in n and not any(nd in n for nd in no_decay)], 
            'weight_decay': float(self.config.weight_decay), 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if 'bert' in n and any(nd in n for nd in no_decay)], 
            'weight_decay': 0, 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and not any(nd in n for nd in no_decay)],
             'weight_decay': float(self.config.weight_decay), 'lr': other_lr},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0, 'lr': other_lr}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, eps=float(self.config.adam_epsilon))
        # 创建优化器
        self.scheduler = get_linear_schedule_with_warmup( #创建学习率调度器
            self.optimizer, num_warmup_steps=self.config.warmup_steps, #预热步骤：学习率升高
            num_training_steps=self.config.epoch_size * len(self.trainLoader)) #总训练步数=训练轮数 × 每轮的批次数

    def get_metrices(self, trues, preds):   
        f1_macro = f1_score(y_true=trues, y_pred=preds, average='macro')
        f1_per_class = f1_score(y_true=trues, y_pred=preds, average=None)
        favor = f1_per_class[0]
        against = f1_per_class[1]
        neutral = f1_per_class[2]
        f1_avg = (favor + against) / 2
        accuracy = accuracy_score(y_true=trues, y_pred=preds)
        return f1_macro, favor, against, neutral, f1_avg, accuracy

    def forward(self):
        self.trainLoader, self.devLoader, self.testLoader = DataProcessor(self.config).get_data()
        #训练集、验证集、测试集
        self.model = SITCL(self.config).to(self.config.device) #创建模型
        self.load_param()
        logging.info('Start training...')
        self.train()
        logging.info('End training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--other_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cuda_index', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--time', type=str, default='')
    parser.add_argument('--gru_layer', type=int, default=1)
    parser.add_argument('--gru_hidden', type=int, default=768)
    parser.add_argument('--model_type', type=str, default='SITPCL') # 文章提出的方法
    parser.add_argument('--plm', type=str, default='roberta')
    parser.add_argument('--bert_dir', type=str, default='/data/nxb/ZS-CSD/plm/chinese-roberta-wwm-ext/')
    args = parser.parse_args()  #args运行参数包
    main = Main(args)
    main.forward()
