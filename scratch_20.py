import pandas as pd
import numpy as np


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

# Разделение df по названию блоков и смена название столбцов в них
#
INDEX = blocks[(blocks[0] == ('//HANDLE'))].index

Support = blocks[(blocks.loc[:,1] == ('опора_промежуточная 0.4'))].reset_index(drop=True)
Support.columns = blocks.loc[INDEX[0]]
Support = Support.loc[:, ['NAME_BLOCK','POINT','N_PYLON']]

light = blocks[(blocks.loc[:,1] == ('ЭО_Опора_1_свет'))].reset_index(drop=True)
light.columns = blocks.loc[INDEX[1]]
light = light.loc[:, ['NAME_BLOCK','POINT','POWER',]]

def points(df):
    for i in range(len(df)):
        xy = df.loc[i,'POINT'].strip('(').strip(')').split()
        df.loc[i,'POINTx'] = float(xy[0])
        df.loc[i,'POINTy'] = float(xy[1])
    return df

def check_and_add(df,index):
    g = 1
    while df.loc[index[0], f'light{g}'] != 0:
        g = g+1
    print(g)
    return g

light = points(light)
Support = points(Support)

number_light = 0
power_light = 0
for i in range(len(Support)):
    x = Support.loc[i, 'POINTx']
    y = Support.loc[i, 'POINTy']
    ss = light[(light.POINTx == x)&(light.POINTy == y)]
    power = ss.loc[:, 'POWER']
    for p in range(len(power)):
        Support.loc[i, f'light{p+1}']= int(power[power.index[p]].strip('Вт'))
        light.loc[power.index[p],'POWER1']=int(power[power.index[p]].strip('Вт'))
        power_light = power_light + int(power[power.index[p]].strip('Вт'))
        number_light = number_light +1

ex = light.isnull()
ex = ex[ex.POWER1 == True]
print(light.loc[ex.index,['POINTx','POINTy','POWER1']])
print(f'Количество светильников {number_light}')
print(f'Мощность всех светильников {power_light}')
result = Support.drop(['NAME_BLOCK', 'POINT','POINTx','POINTy','POINTx'], axis=1)
result = result.sort_values(by=['N_PYLON'])
result = result.set_index('N_PYLON')
result.to_excel('Ведомость.xlsx',sheet_name='0')
