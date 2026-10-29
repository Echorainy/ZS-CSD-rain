from transformers import AutoTokenizer #分词器
import json
from tqdm import tqdm #进度条
from torch.utils.data import DataLoader, Dataset #取数据
import torch

class MyDataset(Dataset): #提供样本的数据集
    def __init__(self, data):
        self.data = data
    def __len__(self): #数据集有多少条样本
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
class DataProcessor(): #读取数据、分词、把数据整理成模型可用的形式
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_dir) #分词器
        self.config = config
    
    def read_data(self, mode):
        if mode == 'train':
            file_path = self.config.train_path
        elif mode == 'dev':
            file_path = self.config.dev_path
        elif mode == 'test':
            file_path = self.config.test_path
        content = json.load(open(file_path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode) #传入对话
            res.append(new_dialog)
        return res
    
    def parse_dialogue(self, dialogue, mode):
        new_sentences = []  # 存放prompt好的mask输入
        new_label_sen = []  # 存放prompt好的立场输入
        target_indices = [] # 存放立场的索引+[SEP] 目标在每句话token序列中的位置范围
        target = dialogue["target"] #取值
        speakers = dialogue["speakers"]
        for id, sen in enumerate(dialogue['sentences']):
            new_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为[MASK][SEP]'
            if id == len(dialogue['sentences']) - 1: #最后一句
                label_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为{dialogue["label"]}[SEP]' #显示立场
                label_sen_token = self.tokenizer.tokenize(label_sen)
                new_label_sen.append(label_sen_token)
            tokens = self.tokenizer.tokenize(new_sen)
            new_sentences.append(tokens)
            sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]'] # 寻找[SEP]的索引
            target_indices.append([sep_indices[0] + 1, sep_indices[1]])
        dialogue['sentences'] = new_sentences
        dialogue['label_sen'] = new_label_sen
        dialogue['target_idx'] = target_indices
        return dialogue
    
    def transform2indices(self, data):
        res = [] 
        for document in data:
            sentences, speakers, label, target_idx, target, label_sen, doc_id = [document[w] for w in ['sentences', 'speakers', 'label','target_idx', 'target', 'label_sen', 'id']]
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, sentences))
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids] # 把输入分为：tokens索引，全1列表，全0列表
            input_ids_label = list(map(self.tokenizer.convert_tokens_to_ids, label_sen))
            input_masks_label = [[1] * len(w) for w in input_ids_label]
            input_segments_label = [[0] * len(w) for w in input_ids_label]
            res.append((input_ids, input_masks, input_segments, speakers, label, target_idx, target, input_ids_label, input_masks_label, input_segments_label, doc_id))
        return res
    
    def forward(self, mode):
        data = self.read_data(mode)
        res = self.transform2indices(data)
        return res
    
    def collate_fn_new(self, batch):
        input_ids, input_masks, input_segments, speakers, label, target_idx, target, input_ids_label, input_masks_label, input_segments_label, doc_id = zip(*batch)
        dialogue_length = list(map(len, input_ids)) # 记录了每个对话的子对话数量
        st = 0
        # input_ids: bz * 每个对话的所有子对话
        dia_idx = [] # 这里是记录了每个对话的子对话索引
        for num in dialogue_length:
            dia_idx.append([st, st + num])
            st += num
        max_lens = max(len(w) for sublist in input_ids for w in sublist) # 找出所有子对话中的最大长度
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for sublist in input_batch for w in sublist]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])
        max_lens_label = max(len(w) for sublist in input_ids_label for w in sublist)
        padding_label = lambda input_batch: [w + [0] * (max_lens_label - len(w)) for sublist in input_batch for w in sublist]
        input_ids_label, input_masks_label, input_segments_label = map(padding_label, [input_ids_label, input_masks_label, input_segments_label]) 
        res = {
            "input_ids": torch.tensor(input_ids).to(self.config.device),
            "input_masks": torch.tensor(input_masks).to(self.config.device),
            "input_segments": torch.tensor(input_segments).to(self.config.device),
            "speakers": speakers,
            "label": torch.tensor(label).to(self.config.device),
            "dia_idx": dia_idx,
            "target_idx": target_idx,
            "target": target,
            "input_ids_label": torch.tensor(input_ids_label).to(self.config.device),
            "input_masks_label": torch.tensor(input_masks_label).to(self.config.device),
            "input_segments_label": torch.tensor(input_segments_label).to(self.config.device),
            "doc_id": doc_id # 对话的样本索引
        }
        return res
    
    def get_data(self):
        if self.config.debug:
            train_dataset = MyDataset(self.forward('train'))[:100]
            dev_dataset = MyDataset(self.forward('dev'))[:100]
            test_dataset = MyDataset(self.forward('test'))[:100]
        else: #默认
            train_dataset = MyDataset(self.forward('train'))
            dev_dataset = MyDataset(self.forward('dev'))
            test_dataset = MyDataset(self.forward('test'))
        train_loader = DataLoader(train_dataset, batch_size=self.config.batchsize, shuffle=True, collate_fn=self.collate_fn_new)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batchsize, shuffle=False, collate_fn=self.collate_fn_new)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batchsize, shuffle=False, collate_fn=self.collate_fn_new)
        return train_loader, dev_loader, test_loader
