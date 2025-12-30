#! -*- coding: utf-8 -*-
# @Time    : 2025/10/1 8:10
# @Author  : LiuGan
import json
import networkx as nx
import networkx.algorithms as algos
import  os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
folder_path = "../data/allData"
all_items = os.listdir(folder_path)
graph = nx.DiGraph()
nerg = set()
setcfc = set()
setysst = set()
setxlms = set()
cfc = []
ysst = []
xlms = []
for fn in all_items:

    filename = "../data/allData/"+fn
    print(filename)

    with open(filename, 'r', encoding='utf-8') as file:
        Privacydata = json.load(file)

    PrivacyGraph = Privacydata['隐私关联']
    print(PrivacyGraph)


    #graph.nodes

    for pg in PrivacyGraph:
        #print(pg)
        graph.add_node(pg['搭配词'])
        cfc.append(pg['搭配词'])
        setcfc.add(pg['搭配词'])
        graph.add_node(pg['隐私实体'])
        ysst.append(pg['隐私实体'])
        setysst.add(pg['隐私实体'])
        graph.add_edge(pg['搭配词'],pg['隐私实体'],relationship=pg['泄露模式'])
        nerg.add(pg['泄露模式'])
        xlms.append(pg['泄露模式'])

print(graph)
print(graph.nodes)
print(graph.edges(data=True))
print(nerg)
print(len(nerg))
"""社区检测"""

naive = algos.community.naive_greedy_modularity_communities(graph)
print(list(naive))
#ga = GraphAnalyzer(graph)