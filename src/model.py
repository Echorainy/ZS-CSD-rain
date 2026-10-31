import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from src.common import map_sequence, target_CL

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values):
        # query: (B, D) or (D,) 查询条件
        # keys/values: (L, D) 由c或k变换得到
        if query.dim() == 1:
            query = query.unsqueeze(0) #（D, ) -> (1, D)
        q = self.query_proj(query)         # (B, D) 投影
        k = self.key_proj(keys)            # (L, D)
        v = self.value_proj(values)        # (L, D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, L) 当前 query 分别与L个Key的匹配分数
        attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, L) 匹配分数->权重
        output = torch.matmul(attn_weights, v)                           # (1, L)*(L, 768) = (B, D) 用权重汇总value
        return output.squeeze(0) if output.size(0) == 1 else output #(1, 768) -> (768,)

class SSE(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.linear_intra = nn.Linear(hidden_dim * 2, hidden_dim)  #（输入维数，输出维数）
        self.linear_inter = nn.Linear(hidden_dim, hidden_dim)
        self.attention_intra = Attention(hidden_dim)
        self.attention_inter = Attention(hidden_dim)

    def forward(self, utterances, speakers):
        # utterances: (L, D) 当前所有对话经GRU处理后的向量 eg：utterances = [u0, u1, u2, u3]，每句话对对应一个向量u[i],实际utterances.shape = (4, 768)
        # speakers: list or tensor of L
        device = utterances.device
        speakers = torch.tensor(map_sequence(speakers), device=device) # 把对话人id投影为等差数列，并按出现顺序排列: [0,2,5,2,5,2]->[0,1,2,1,2,1]
        V_lst = [] #保存每句话经过 SSE 处理后的结果
        last_speaker_idx = dict() #（说话者编号，最后一次发言位置）
        for i in range(len(speakers)): # 每个轮次
            speaker_id = speakers[i].item() # 张量 → 数字 
            if speaker_id not in last_speaker_idx:
                V_lst.append(utterances[i]) # 第一次说话，直接保存当前GRU输出
            else:
                prev_idx = last_speaker_idx[speaker_id] # 当前对话人上一次的对话轮次
                vh_concat = torch.cat((V_lst[prev_idx], utterances[i]), dim=-1) # 把同一个人说的子对话按照最后一维向量维度拼接768 -> 768*2 (D*2)
                	# cat：沿已有维度拼接。 stack：新增一个维度，把张量堆在一起。
                q_intra = self.linear_intra(vh_concat) # 768*2 -> 768 (D)把同一说话者上一次发言的SSE表示和当前发言的GRU表示拼接，经linear得的查询向量
                c = utterances[:i+1] # 之前所有子对话+当前对话人的子对话的GRU结果 截至当前句i的向量
                v_intra = self.attention_intra(q_intra, c, c) # 计算注意力，得到(D) 在查询向量引导下，对截至当前的对话表示进行加权汇总，得到当前话语的表示

                q_inter = self.linear_inter(utterances[i]) # (D) 当前话语的GRU表示
                k = torch.stack([V_lst[j] for j in range(prev_idx, i)]) if i > prev_idx else utterances[i].unsqueeze(0) # k是[当前对话人的上一个子对话,上一个对话人的子对话]
                v_inter = self.attention_inter(q_inter, k, k) if len(k) > 0 else torch.zeros_like(q_inter)

                V_lst.append(v_intra + v_inter) # 在同维度上相加
            last_speaker_idx[speaker_id] = i # 更新最后一个对话人的轮次
        return torch.stack(V_lst)

class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.alpha
        self.bert = AutoModel.from_pretrained(config.bert_dir) #加载预训练语言模型，向量表示
        self.gru = nn.GRU(input_size=768, #输入一条对话所需向量的维度 
        	hidden_size=config.gru_hidden, #GRU隐藏状态的维度 
			num_layers=config.gru_layer #GRU层数
            ,batch_first=True)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes) #nn.Linear(输入维度, 输出维度) 话语转换成各类分数
        self.criterion = nn.CrossEntropyLoss()#创建交叉墒损失函数
        self.SSE = SSE(hidden_dim=config.gru_hidden)

    def forward(self, **kwargs): #kwargs：写入字典
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        targets = kwargs['target']
        # 所有单对话*seq_len*768
        out = self.bert(input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments).last_hidden_state
vx
        H_final = [] #经SSE处理后的对话
        stance = []#每段对话最后一句的立场
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = out[st:ed, -2, :] # 取单个对话（当前对话的所有子对话）的向量，(L,D)，取每个字对话的倒数第二个token位置，保留该位置向量中全部768个数
            o, _ = self.gru(h.unsqueeze(0))      # (1, L, D) 当前对话经过 GRU 后的各句话表示。（一条对话，L句话，D维）
            o = o.squeeze(0)                     # (L, D)
            v = self.SSE(o, speakers[dia_id]) # Speaker Interaction Encoder
            H_final.append(v)
            stance.append(v[-1])  # 取最后一个子对话作为立场
        stance = torch.stack(stance)             # (B, D) 把形状相同的张量沿着一个新维度堆叠起来
        logits = self.fc(stance)                 # (B, num_classes)把每个向量变为三个立场分数
        ce_loss = self.criterion(logits, label) #分类损失
        target_contrastive_loss = target_CL(H_final, targets, self.config) #对比损失
        loss = ce_loss + self.alpha * target_contrastive_loss
        return loss, logits, label 