import pandas as pd
import numpy as np
import math
# import pymysql
# import sqlalchemy





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

Support = blocks[(blocks.loc[:, 1] == ('опора_промежуточная 0.4'))].reset_index(drop=True)
Support.columns = blocks.loc[INDEX[0]]
Support = Support.loc[:, ['NAME_BLOCK', 'POINT', 'N_PYLON']]

light = blocks[(blocks.loc[:, 1] == ('ЭО_Опора_1_свет'))].reset_index(drop=True)
light.columns = blocks.loc[INDEX[1]]
light = light.loc[:, ['NAME_BLOCK', 'POINT', 'POWER', 'Angle1']]


def points(df):
    for i in range(len(df)):
        xy = df.loc[i, 'POINT'].strip('(').strip(')').split()
        df.loc[i, 'POINTx'] = float(xy[0])
        df.loc[i, 'POINTy'] = float(xy[1])
    return df


light = points(light)
light = light.assign(POWER=light.POWER.str.replace(r'Вт$', ''))
light.POWER = light.POWER.astype('float')
light.Angle1 = light.Angle1.astype('float')
light.Angle1 = light.Angle1 / math.pi * 180
Support = points(Support)

def get_distance(unit1, unit2):
    phi = abs(unit2-unit1) % 360
    sign = 1
    # used to calculate sign
    if not ((unit1-unit2 >= 0 and unit1-unit2 <= 180) or (
            unit1-unit2 <= -180 and unit1-unit2 >= -360)):
        sign = -1
    if phi > 180:
        result = 360-phi
    else:
        result = phi
    return abs(int(round(result*sign/5.0)*5.0))


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
    return Support, power_light, number_light, light


result, power_light, number_light, light = light_in_support(Support, light)

ex = light.isnull()
ex = ex[ex.POWER1 == True]
print(light.loc[ex.index, ['POINTx', 'POINTy', 'POWER']])
print(f'Количество светильников {number_light}')
print(f'Мощность всех светильников {power_light}')
result = result.drop(['NAME_BLOCK', 'POINT', 'POINTx', 'POINTy', 'POINTx'], axis=1)
result = result.sort_values(by=['N_PYLON'])
# result = result.set_index('N_PYLON')
result.to_excel('Ведомость.xlsx', sheet_name='0')

# import pymysql
#
# #database connection
# connection = pymysql.connect(host="127.0.0.1", user="Fedor", passwd="fedor", port=3306,database="blocks")
# cursor = connection.cursor()
#
# # queries for inserting values
# insert1 = """INSERT INTO fine(name,number_plate,violation,sum_fine,date_violation,date_payment)
# VALUES('Баранов П.Е.','Р523ВТ','Превышение скорости(от 40 до 60)',Null,'2020-02-14',Null),
# ('Абрамова К.А.','О111АВ','Проезд на запрещающий сигнал',Null,'2020-02-23',Null),
# ('Яковлев Г.Р.','Т330ТТ','Проезд на запрещающий сигнал',Null,'2020-03-03',Null);"""
#
# #executing the quires
# cursor.execute(insert1)
#
#
#
# #commiting the connection then closing it.
# connection.commit()
# connection.close()

# #database connection
# connection = pymysql.connect(host="127.0.0.1", user="root", passwd="root", port=3306,database="blocks")
# cursor = connection.cursor()
# host_ip = "127.0.0.1:3306"
#
# #создаю 2 таблицы в базе данных
# # engine = sqlalchemy.create_engine('mysql://mysql:root@%s:3306' % host_ip)
# engine = sqlalchemy.create_engine('mysql+pymysql://root:root@127.0.0.1:3306/blocks')
# result.to_sql('result', engine, schema=None, if_exists='replace', index=False)
# blocks.to_sql('blocks', engine, schema=None, if_exists='replace', index=False)
