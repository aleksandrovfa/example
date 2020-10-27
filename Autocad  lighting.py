import pandas as pd
import numpy as np

# import data
light = pd.read_excel('light.xlsx', header=0)
Support = pd.read_excel('Support.xlsx', header=0)

light = light[(light.NAME_BLOCK == ('ЭО_Опора_1_свет'))]
light = light.loc[:, ['NAME_BLOCK', 'POINT', 'POWER']]

Support = Support[(Support.NAME_BLOCK == ('опора_промежуточная 0.4'))]
Support = Support.loc[:, ['NAME_BLOCK', 'POINT', 'N_PYLON']]


def points(df):
    for i in range(len(df)):
        xy = df.loc[i, 'POINT'].strip('(').strip(')').split()
        df.loc[i, 'POINTx'] = float(xy[0])
        df.loc[i, 'POINTy'] = float(xy[1])
    return df


def check_and_add(df, index):
    g = 1
    while df.loc[index[0], f'light{g}'] != 0:
        g = g + 1
    print(g)
    return g


light = points(light)
Support = points(Support)

number_light = 0
power_light = 0
for i in range(len(Support)):
    x = Support.loc[i, 'POINTx']
    y = Support.loc[i, 'POINTy']
    ss = light[(light.POINTx == x) & (light.POINTy == y)]
    power = ss.loc[:, 'POWER']
    for p in range(len(power)):
        Support.loc[i, f'light{p + 1}'] = int(power[power.index[p]].strip('Вт'))
        light.loc[power.index[p], 'POWER1'] = int(power[power.index[p]].strip('Вт'))
        power_light = power_light + int(power[power.index[p]].strip('Вт'))
        number_light = number_light + 1

ex = light.isnull()
ex = ex[ex.POWER1 == True]
print(light.loc[ex.index, ['POINTx', 'POINTy', 'POWER1']])
print(f'Количество светильников {number_light}')
print(f'Мощность всех светильников {power_light}')
result = Support.loc[:, ['N_PYLON', 'POINTx', 'POINTy', 'light1', 'light2', 'light3']]
result = result.sort_values(by=['N_PYLON'])
result = result.set_index('N_PYLON')
result.to_excel('Ведомость.xlsx', sheet_name='0')
