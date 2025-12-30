#! -*- coding: utf-8 -*-
# @Time    : 2025/12/3 16:03
# @Author  : LiuGan
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import numpy as np
# ============================
# 1. 示例三元组数据 (E, C, T)
# ============================
with open("../deepseekAPI/data4.json","r",encoding="utf-8") as f:
    data=json.load(f)


triplets = data
# triplets = [
#     ("南京大学", "就读于", "教育信息泄露"),
#     ("华为公司", "任职于", "工作单位泄露"),
#     ("上海市浦东新区", "居住在", "地址泄露"),
#     ("北京大学", "毕业于", "教育信息泄露"),
#     ("阿里巴巴集团", "工作于", "工作单位泄露"),
#     ("北京市海淀区", "户籍所在地", "地址泄露"),
# ]


# ===================================
# 2. 构造 Dataset：正样本(E,C) + 负样本
# ===================================
class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = list({t[2] for t in data})

        # 按 T 分组
        self.by_T = {}
        for e, c, t in data:
            self.by_T.setdefault(t, []).append((e, c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        E, C, T = self.data[idx]

        # 正样本：同类 T 的 (E, C)
        anchor_text = E
        positive_text = C

        # 负样本：从不同 T 中随机选一个实体或触发词
        neg_T = random.choice([t for t in self.by_T.keys() if t != T])
        neg_E, neg_C = random.choice(self.by_T[neg_T])

        # 随机使用实体或触发词作为负样本
        negative_text = random.choice([neg_E, neg_C])

        return anchor_text, positive_text, negative_text


# =============================
# 3. BERT 编码器 + 投影层
# =============================
class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("./BERT")
        self.proj = nn.Linear(768, 128)  # 映射到低维空间

    def forward(self, texts):
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.bert(**tokens)
        cls_embed = outputs.last_hidden_state[:, 0]  # [CLS]

        return self.proj(cls_embed)  # 投影后的特征向量


# ======================
# 4. Triplet Loss
# ======================
class TripletLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertEncoder()
        self.loss_fn = nn.TripletMarginLoss(margin=1.0)

    def forward(self, anchor, positive, negative):
        a = self.encoder(anchor)
        p = self.encoder(positive)
        n = self.encoder(negative)
        loss = self.loss_fn(a, p, n)
        return loss


# ======================
# 5. 启动训练
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
tokenizer = BertTokenizer.from_pretrained("./BERT")

dataset = TripletDataset(triplets)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TripletLossModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# ======================
# 6. 训练循环
# ======================
for epoch in range(1):
    total_loss = 0
    for anchor, pos, neg in loader:
        loss = model(anchor, pos, neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss = {total_loss:.4f}")

print("训练完成！")


# ===============================
# 7. 测试：比较特征空间距离
# ===============================

list0,list1,list2,list3,list4,list5,list6 = [],[],[],[],[],[],[]
with open("../deepseekAPI/data4.json", "r",encoding="utf-8") as f:
    data = json.load(f)
labellist = ["机构(公司)名", "地址信息", "职业信息", "教育水平", "年龄信息", "兴趣信息", "个人基本信息"]

for i in data:
    if "机构(公司)名" in i:
        list0.append(i)
    if "地址信息" in i:
        list1.append(i)
    if "职业信息" in i:
        list2.append(i)
    if "教育水平" in i:
        list3.append(i)
    if "年龄信息" in i:
        list4.append(i)
    if "兴趣信息" in i:
        list5.append(i)
    if "个人基本信息" in i:
        list6.append(i)

#print(len(list0),len(list1),len(list2),len(list3),len(list4),len(list5),len(list6))

data1 = random.sample(list0, 40)
data2 = random.sample(list1, 40)
data3 = random.sample(list2, 40)
data4 = random.sample(list2, 40)
data5 = random.sample(list2, 40)
data6 = random.sample(list3, 40)
data7 = random.sample(list4, 40)
data8 = random.sample(list5, 40)
data9 = random.sample(list6, 40)
datalist0,datalist1,datalist2,datalist3,datalist4,datalist5,datalist6,datalist7,datalist8 = [],[],[],[],[],[],[],[],[]
yencode = BertEncoder().to(device)
def cosine_sim(a, b):
    return torch.cosine_similarity(a, b).item()

def rmodel(a):
    with torch.no_grad():
        b=yencode(a)
        b=b.cpu()
        return b.numpy().tolist()
def ymodel(a):
    with torch.no_grad():
        b=model.encoder(a)
        b=b.cpu()
        return b.numpy().tolist()
for i in data1:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist0.append(cunlist)

for i in data2:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist1.append(cunlist)

for i in data3:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist2.append(cunlist)

for i in data4:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist3.append(cunlist)

for i in data5:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist4.append(cunlist)

for i in data6:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist5.append(cunlist)

for i in data7:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist6.append(cunlist)

for i in data8:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist7.append(cunlist)

for i in data9:
    cunlist=[]
    print(i[0])
    print(i[1])
    Rsc=rmodel([i[0]])
    cunlist.append(Rsc)
    Rse=rmodel([i[1]])
    cunlist.append(Rse)
    Psc= ymodel([i[0]])
    cunlist.append(Psc)
    Pse= ymodel([i[1]])
    cunlist.append(Pse)
    datalist8.append(cunlist)

with open('./filework/listdata0.json', 'w',encoding='utf-8') as f3:
    json.dump(datalist0, f3, ensure_ascii=False, indent=4)

with open('./filework/listdata1.json', 'w',encoding='utf-8') as f4:
    json.dump(datalist1, f4, ensure_ascii=False, indent=4)

with open('./filework/listdata2.json', 'w',encoding='utf-8') as f5:
    json.dump(datalist2, f5, ensure_ascii=False, indent=4)

with open('./filework/listdata3.json', 'w',encoding='utf-8') as f6:
    json.dump(datalist3, f6, ensure_ascii=False, indent=4)

with open('./filework/listdata4.json', 'w',encoding='utf-8') as f7:
    json.dump(datalist4, f7, ensure_ascii=False, indent=4)

with open('./filework/listdata5.json', 'w',encoding='utf-8') as f8:
    json.dump(datalist5, f8, ensure_ascii=False, indent=4)

with open('./filework/listdata6.json', 'w',encoding='utf-8') as f9:
    json.dump(datalist6, f9, ensure_ascii=False, indent=4)

with open('./filework/listdata7.json', 'w',encoding='utf-8') as f10:
    json.dump(datalist7, f10, ensure_ascii=False, indent=4)

with open('./filework/listdata8.json', 'w',encoding='utf-8') as f11:
    json.dump(datalist8, f11, ensure_ascii=False, indent=4)







# test_E = "山东英大科技有限公司"
# test_C = "董事长"
#
# with torch.no_grad():
#
#     Y_E = yencode([test_E])
#     Y_C = yencode([test_C])
#     print(Y_E)
#     print(Y_C)
# print("北京大学 vs 就读于 的相似度：", cosine_sim(Y_E, Y_C))
# with torch.no_grad():
#     e_vec = model.encoder([test_E])
#     c_vec = model.encoder([test_C])
#     print(e_vec)
#     print(c_vec)
#
# print("北京大学 vs 就读于 的相似度：", cosine_sim(e_vec, c_vec))
