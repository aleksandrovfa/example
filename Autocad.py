import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Опорный узел, с которого начинается расчет (источник)
Support_node = 'N'

cable_df =pd.DataFrame(np.array([['ЭО_Воздушная линия', 'СИП-2А-3x25+1x35', 25, 35, 13.1, 1.47, 1.05],
                                 ['ЭО_Кабельная линия', 'ПвВГнг(А)-4х16', 16, 16, 21.9, 1.37, 1.37],
                                 ]),
                   columns=['Type', 'NameCable',  'Sf', 'So', 'alfa', 'Rf', 'Ro'])
#ЭО_Воздушная_линия- СИП-2А-3x25+1x35  alfa =13,1  Rf=1,47 Ro = 1,05

#ЭО_Кабельная линия- ПвВГнг(А)-4х16    alfa =21,1  Rf=1,37 Ro = 1,37


# Функции для поэтапной обработки, в целом названия говорят сами за себя.
# Честно сказать оформление очень херовенькое, весь скрипт тупо написание отдельных функций и поэтапная обработка
def of_TXT_in_df(file):
    file = file.read().rstrip().split('\n')
    g = []
    for i in file:
        lst = i.split('\t')
        g.append(lst)
    file = pd.DataFrame(g)
    return file


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


def get_distance(unit1, unit2):
    phi = abs(unit2 - unit1) % 360
    sign = 1
    # used to calculate sign
    if not ((0 <= unit1 - unit2 <= 180) or
            (-180 >= unit1 - unit2 >= -360)):
        sign = -1
    if phi > 180:
        result = 360 - phi
    else:
        result = phi
    return abs(int(round(result * sign / 5.0) * 5.0))


def light_in_support(Support, light):
    number_light = 0
    power_light = 0
    for index, row in Support.iterrows():
        ss = light[(light.POINTx == row.POINTx) & (light.POINTy == row.POINTy)]
        power = ss.loc[:, ['POWER', 'Angle1']].sort_values(by=['POWER'], ascending=False)
        Support.loc[index, 'Count'] = int(len(power))
        Support.loc[index, 'Power'] = power.POWER.sum()
        if len(power) < 2:
            Support.loc[index, 'Angle'] = 0
        elif len(power) == 2:
            Support.loc[index, 'Angle'] = get_distance(power.Angle1[power.index[0]], power.Angle1[power.index[1]])
        elif len(power) == 3:
            angle1 = get_distance(power.Angle1[power.index[0]], power.Angle1[power.index[1]])
            angle2 = get_distance(power.Angle1[power.index[0]], power.Angle1[power.index[2]])
            Support.loc[index, 'Angle'] = min(angle1, angle2)
        for p in range(len(power)):
            Support.loc[index, f'light{p + 1}'] = power.POWER[power.index[p]]
            light.loc[power.index[p], 'POWER1'] = power.POWER[power.index[p]]
            power_light = power_light + power.POWER[power.index[p]]
            number_light = number_light + 1
    light1 = light.isnull()
    light1 = light1[light1.POWER1 == True]
    light = light.loc[light1.index, ['POINTx', 'POINTy', 'POWER']]
    print(light)
    print(f'Количество светильников {number_light}')
    print(f'Мощность всех светильников {power_light}')
    return Support


def cable_in_support(line, Support):
    for index, row in line.iterrows():
        try:
            a = Support[(Support.POINTx == row.POINTx1) & (Support.POINTy == row.POINTy1)].N_PYLON
            b = Support[(Support.POINTx == row.POINTx2) & (Support.POINTy == row.POINTy2)].N_PYLON
            line.loc[index, 'N_PYLON1'] = str(a[a.index[0]])
            line.loc[index, 'N_PYLON2'] = str(b[b.index[0]])
        except:
            print('Не найдена опора в линии между ', line.loc[index, 'N_PYLON1'], line.loc[index, 'N_PYLON2'])
        print('Привязка линии длиной', line.loc[index, 'LENGTH'], 'слоя', line.loc[index, 'LAYER'], 'между',
              line.loc[index, 'N_PYLON1'], line.loc[index, 'N_PYLON2'])
    line['LENGTH'] = line['LENGTH'].astype('float')
    return line


def draw_graph(G):
    plt.figure()
    pos = nx.kamada_kawai_layout(G)
    edges = G.edges()
    colors = [G[u][v][0]['LAYER'] for u, v in edges]
    color = []
    for i in colors:
        if i == 'ЭО_Кабельная линия':
            color.append('g')
        if i == 'ЭО_Воздушная линия':
            color.append('r')
    nx.draw(G, pos, edge_color=color, with_labels=True)
    plt.show(block=False)
    # print(nx.dfs_successors(G,"N"))
    # print(nx.dfs_predecessors(G,"N"))
    # print(G["N"])


def draw_graph1(Ga, Support):
    Support = Support.set_index('N_PYLON')
    plt.figure()
    pos = nx.kamada_kawai_layout(Ga)
    nx.draw_networkx_nodes(Ga, pos, node_color='b', node_size=150, alpha=0.2, label=True)
    nx.draw_networkx_labels(Ga, pos, font_size=10)
    ax = plt.gca()
    for e in Ga.edges:
        color = nx.get_edge_attributes(Ga, 'LAYER')
        if color[e] == 'ЭО_Кабельная линия':
            color = 'g'
        elif color[e] == 'ЭО_Воздушная линия':
            color = 'r'
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color,
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.2 * e[2])
                                                                           ),
                                    ),
                    )
        ax.text(pos[e[1]][0] - 0.03, pos[e[1]][1] - 0.04, Support.loc[e[1], 'GROUP'],
                fontsize=7)
    plt.axis('off')
    plt.show()
    plt.show()


def support_selection(G, Support):
    Support = Support.set_index('N_PYLON')
    for N_PYLON in Support.index:
        weight_x, weight_y = 0, 0
        coefficient = 0
        x1 = int(Support[Support.index == N_PYLON].POINTx)
        y1 = int(Support[Support.index == N_PYLON].POINTy)
        for CN_PYLON in G[N_PYLON]:
            for num in G[N_PYLON][CN_PYLON]:
                if G[N_PYLON][CN_PYLON][num]['LAYER'] == 'ЭО_Воздушная линия':
                    x2 = int(Support[Support.index == CN_PYLON].POINTx)
                    y2 = int(Support[Support.index == CN_PYLON].POINTy)
                    delt_x = (x2 - x1) / 2
                    delt_y = (y2 - y1) / 2
                    weight_x = weight_x + delt_x
                    weight_y = weight_y + delt_y
        coefficient = math.sqrt((weight_x ** 2) + (weight_y ** 2))
        Support.loc[N_PYLON, 'PYLON'] = coefficient
        if coefficient == 0:
            Support.loc[N_PYLON, 'PYLON_Type'] = "Обычная"
        elif 0 < coefficient < 7500:
            Support.loc[N_PYLON, 'PYLON_Type'] = "Проходная"
        elif 7500 <= coefficient < 17500:
            Support.loc[N_PYLON, 'PYLON_Type'] = "Концевая"
        elif 17500 <= coefficient:
            Support.loc[N_PYLON, 'PYLON_Type'] = "Угловая"
    Support = Support.reset_index()
    return Support


def support_set_power(G, Support):
    Support = Support.set_index('N_PYLON')
    tree = nx.dfs_successors(G, Support_node)
    for GROUP in Support['GROUP'].unique():
        Support[f'{GROUP}'] = Support[Support.GROUP == GROUP].Power
        Support[f'{GROUP}'] = Support[f'{GROUP}'].fillna(0)
        for i in reversed(list(tree.keys())):
            for g in tree[i]:
                Support.loc[i, f'{GROUP}'] = Support.loc[i, f'{GROUP}'] + Support.loc[g, f'{GROUP}']
    Support = Support.reset_index()
    return Support


def cable_list(G, Support, Support_node):
    Support = Support.set_index('N_PYLON')
    cable_list = list(nx.dfs_edges(G, Support_node))
    cable_list = pd.DataFrame(data=cable_list)
    cable_list = cable_list.rename(columns={0: 'N_PYLON1', 1: 'N_PYLON2'})
    for index, row in cable_list.iterrows():
        Line_row = Line[((Line.N_PYLON1 == row['N_PYLON1']) & (Line.N_PYLON2 == row['N_PYLON2'])) |
                        ((Line.N_PYLON1 == row['N_PYLON2']) & (Line.N_PYLON2 == row['N_PYLON1']))]
        cable_list.loc[index, 'LENGTH'] = Line_row.LENGTH.values
        cable_list.loc[index, 'LAYER'] = Line_row.LAYER.values
        for GROUP in Support['GROUP'].unique():
            cable_list.loc[index, f'{GROUP}'] = Support.loc[row['N_PYLON2'], GROUP]
    cable_list1 = pd.DataFrame()
    for GROUP in Support['GROUP'].unique():
        df = cable_list[cable_list[f'{GROUP}'] != 0].loc[:, ('N_PYLON1', 'N_PYLON2', 'LENGTH', 'LAYER', f'{GROUP}')]
        df = df.rename(columns={f'{GROUP}': 'POWER'})
        df['GROUP'] = GROUP
        cable_list1 = pd.concat([cable_list1, df])
    return cable_list, cable_list1


# Создание двух df по выгруженным тхт файлам
blocks = open('OUT_block.txt')
blocks = of_TXT_in_df(blocks)
line = open('OUT_line.txt')
line = of_TXT_in_df(line)

# Обработка df, разделение на df Support,light,Line
# Line - кабельный журнал
# Support - Ведомость опор
INDEXblocks = blocks[(blocks[0] == '//HANDLE')].index
Support = blocks[(blocks.loc[:, 1] == 'опора_промежуточная 0.4')].reset_index(drop=True)
Support.columns = blocks.loc[INDEXblocks[0]]
Support = Support.loc[:, ['NAME_BLOCK', 'POINT', 'N_PYLON', 'GROUP']]

#создание df по светильникам
light = blocks[(blocks.loc[:, 1] == 'ЭО_Опора_1_свет')].reset_index(drop=True)
light.columns = blocks.loc[INDEXblocks[1]]
light = light.loc[:, ['NAME_BLOCK', 'POINT', 'POWER', 'Angle1']]

#создание df по линиям
INDEXline = line[(line[0] == '//HANDLE')].index
line.columns = line.loc[INDEXline[0]]
Line = line[(line['LAYER'] == 'ЭО_Воздушная линия') | (line['LAYER'] == 'ЭО_Кабельная линия')].reset_index(drop=True)

light = points(light)
light = light.assign(POWER=light.POWER.str.replace(r'Вт$', ''))
light.loc[:, ['POWER', 'Angle1']] = light.loc[:, ['POWER', 'Angle1']].astype('float')
light.Angle1 = light.Angle1 / math.pi * 180

# Началась обработка(форматирование координат)
Support = points(Support)
Line = points_line1(Line)

Support = light_in_support(Support, light)
Line = cable_in_support(Line, Support)

G = nx.from_pandas_edgelist(Line, 'N_PYLON1', 'N_PYLON2', ['LENGTH', 'LAYER'], create_using=nx.MultiGraph)
Support = support_set_power(G, Support)
cable_list, cable_list1 = cable_list(G, Support, Support_node)

G = nx.from_pandas_edgelist(cable_list1, 'N_PYLON1', 'N_PYLON2', ['LENGTH', 'LAYER'], create_using=nx.MultiGraph)
Support = support_selection(G, Support)
draw_graph1(G, Support)

resultSupport = Support.drop(['NAME_BLOCK', 'POINT', 'POINTx', 'POINTy', 'POINTx'], axis=1)
resultSupport = resultSupport.sort_values(by=['N_PYLON'])
resultSupport.to_excel('Ведомость.xlsx', sheet_name='0')

# resultLine = Line.loc[:, ['N_PYLON1', 'N_PYLON2', 'LENGTH', 'LAYER']]
# resultLine.to_excel('Кабельный журнал.xlsx', sheet_name='0')
resultLine = cable_list.reset_index(drop=True)
resultLine.to_excel('Кабельный журнал.xlsx', sheet_name='0')
resultLine = cable_list1.reset_index(drop=True)
resultLine.to_excel('Кабельный журнал1.xlsx', sheet_name='0')

#
# for num in Ga['N2_60']['N2_13']:
#     print(Ga['N2_60']['N2_13'][num])bmnb,,mnb



for index, row in cable_list1.iterrows():
    cable_list1.loc[index, 'dU_area'] = float(cable_df[cable_df['Type'] == row['LAYER']].alfa) * \
                                        row['POWER']/1000 * row['LENGTH']/(10 ** 3) / \
                                        float(cable_df[cable_df['Type'] == row['LAYER']].Sf)
cable_list1 = cable_list1.reset_index(drop = True)
cable_list1.loc[:, 'dU'] = cable_list1.loc[:, 'dU_area']
for i in range(len(cable_list1)):
    cable = cable_list1[(cable_list1.N_PYLON1 == cable_list1.loc[i,'N_PYLON2']) & (cable_list1.GROUP == row['GROUP'])]
    print(cable)
    if len(cable.index.values) > 0:
        cable_list1.loc[cable.index, 'dU'] = cable_list1.loc[cable.index, 'dU']+ cable_list1.loc[i,'N_PYLON2']




# cable_df[cable_df['Type'] == row['LAYER']].NameCable

# line.loc[index, 'N_PYLON1']