import calendar
import random
from datetime import datetime, timedelta

import pandas as pd

df = pd.read_csv("dataset/oil_tanks/large_image_data.csv")

ano = 2018
mes_sorteado = random.randint(1, 12)
_, ultimo_dia = calendar.monthrange(ano, mes_sorteado)

data_inicial = datetime(ano, mes_sorteado, 1)
data_final = datetime(ano, mes_sorteado, ultimo_dia)

total_dias_intervalo = (data_final - data_inicial).days + 1
datas_aleatorias = []

for _ in range(len(df)):
    dias_aleatorios = random.randint(0, total_dias_intervalo)
    data_aleatoria = data_inicial + timedelta(days=dias_aleatorios)
    datas_aleatorias.append(data_aleatoria.strftime("%Y-%m-%d"))


df["Data"] = datas_aleatorias
df.to_csv("dataset/oil_tanks/large_image_data_with_dates.csv", index=False)
