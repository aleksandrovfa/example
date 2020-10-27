import itertools
import networkx as nx
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def of_TXT_in_df(file):
    file = file.read().rstrip().split('\n')
    g = []
    for i in file:
        lst = i.split('\t')
        g.append(lst)
    file = pd.DataFrame(g)
    return file

# Создание двух df по выгруженным тхт файлам
blocks = open('OUT_block.txt')
blocks = of_TXT_in_df(blocks)
line = open('OUT_line.txt')
line = of_TXT_in_df(line)

INDEXblocks = blocks[(blocks[0] == ('//HANDLE'))].index

Support = blocks[(blocks.loc[:, 1] == ('опора_промежуточная 0.4'))].reset_index(drop=True)
Support.columns = blocks.loc[INDEXblocks[0]]
Support = Support.loc[:, ['NAME_BLOCK', 'POINT', 'N_PYLON']]

INDEXline = line[(line[0] == ('//HANDLE'))].index

LineVL = line[(line.loc[:, 1] == ('ЭО_Воздушная линия'))].reset_index(drop=True)
LineKL = line[(line.loc[:, 1] == ('ЭО_Кабельная линия'))].reset_index(drop=True)
LineVL.columns = line.loc[INDEXline[0]]
LineKL.columns = line.loc[INDEXline[0]]

def points_line(df):
    df = df.loc[:, 'COORDS':]
    df.columns = range(df.shape[1])
    for g in range(df.shape[1]):
        for i in range(df.shape[0]):
            if df.loc[i, g] is not None:
                xy = df.loc[i, g].strip('(').strip(')').split()
                df.loc[i, f'POINTx{g}'] = round(float(xy[1]), -3)
                df.loc[i, f'POINTy{g}'] = round(float(xy[2]), -3)
            else:
                df.loc[i, f'POINTx{g}'] = None
                df.loc[i, f'POINTy{g}'] = None
    return df

def points_line1(line):
    df = line.loc[:, 'COORDS':]
    df.columns = range(df.shape[1])
    for index, row in df.iterrows():
        xy1 = df.loc[index, 0].strip('(').strip(')').split()
        xy2 = df.loc[index, len(row.dropna()) - 1].strip('(').strip(')').split()
        line.loc[index, 'POINTx1'] = round(float(xy1[1]), -3)
        line.loc[index, 'POINTy1'] = round(float(xy1[2]), -3)
        line.loc[index, 'POINTx2'] = round(float(xy2[1]), -3)
        line.loc[index, 'POINTy2'] = round(float(xy2[2]), -3)
    return line

def points(df):
    for i in range(len(df)):
        xy = df.loc[i, 'POINT'].strip('(').strip(')').split()
        df.loc[i, 'POINTx'] = round(float(xy[0]), -3)
        df.loc[i, 'POINTy'] = round(float(xy[1]), -3)
    return df


Support = points(Support)
LineVL = points_line1(LineVL)
LineKL = points_line1(LineKL)

# cable_list
def cable_list(line, Support):
    for index, row in line.iterrows():
        try:
            a = Support[(Support.POINTx == row.POINTx1) & (Support.POINTy == row.POINTy1)].N_PYLON
            b = Support[(Support.POINTx == row.POINTx2) & (Support.POINTy == row.POINTy2)].N_PYLON
            line.loc[index, 'N_PYLON1'] = str(a[a.index[0]])
            line.loc[index, 'N_PYLON2'] = str(b[b.index[0]])
        except :
            print('Не найдена опора в линии между ', line.loc[index, 'N_PYLON1'], line.loc[index, 'N_PYLON2'])
        print('Привязка линии длиной', line.loc[index, 'LENGTH'], 'слоя', line.loc[index, 'LAYER'], 'между',
          line.loc[index, 'N_PYLON1'], line.loc[index, 'N_PYLON2'])
    line['LENGTH']=line['LENGTH'].astype('float')
    return line

LineVL = cable_list(LineVL,Support)
LineKL = cable_list(LineKL,Support)

print(LineVL.loc[:, ['N_PYLON1', 'N_PYLON2', 'LENGTH', 'LAYER']])
print(LineKL.loc[:, ['N_PYLON1', 'N_PYLON2', 'LENGTH', 'LAYER']])

def get_list_to_graf(df):
    df = df.loc[:, ['N_PYLON1', 'N_PYLON2', 'LENGTH']]
    Glist = []
    for index, row in df.iterrows():
        row = row.to_list()
        Glist.append(row)
    return Glist


G = nx.Graph()
# G=nx.from_pandas_dataframe(a,'N_PYLON1','N_PYLON2')
# elist=[('a','b',5.0),('b','c',3.0),('a','c',1.0),('c','d',7.3)]
VLgraf =get_list_to_graf(LineVL)
KLgraf =get_list_to_graf(LineKL)

G.add_weighted_edges_from(VLgraf)
G.add_weighted_edges_from(KLgraf)

nx.draw(G,node_color='red',
         node_size=500,
         with_labels=True)

a = nx.dfs_successors(G,"N2_46")
b = nx.dfs_predecessors(G,"N2_46")
plt.show()


# G = nx.Graph()
# # G=nx.from_pandas_dataframe(a,'N_PYLON1','N_PYLON2')
# # elist=[('a','b',5.0),('b','c',3.0),('a','c',1.0),('c','d',7.3)]
# VLgraf =get_list_to_graf(LineVL)
# KLgraf =get_list_to_graf(LineKL)
#
# G.add_weighted_edges_from(VLgraf)
# G.add_weighted_edges_from(KLgraf)
#
# nx.draw(G,node_color='red',
#          node_size=500,
#          with_labels=True)
# print(nx.dfs_successors(G,"N2_46"))
# plt.show()
