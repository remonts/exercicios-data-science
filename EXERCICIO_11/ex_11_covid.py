# importando biblioteca pandas
import pandas as pd
# imprimir gráfico
from matplotlib import pyplot as plt
# nuvem de palabras
from wordcloud import wordcloud

# variavel que vai receber o dataset
dados = pd.read_csv("covid19_12052020.csv")



# imprimir somente o numero de casos confirmados e mortes do dia data_maxima, por estado
#print(dados_data_maxima[['confirmed', 'deaths']].sum())

# LETRA A
# mostrar colunas data, estado, confirmados e mortes por estado
dados_grafico_1 = dados[dados['place_type'] == 'state']
dados_grafico_1 = dados_grafico_1[['date', 'state', 'confirmed', 'deaths']]
# agrupar os dados por data, exibindo coluna de confirmados
# reset_index para mostrat a coluna date
dados_grafico_1 = dados_grafico_1.groupby(['date'])['confirmed', 'deaths'].sum().reset_index().sort_values(by=['date'])

# imprimindo o gráfico
#dados_grafico_1.plot.line(x='date')
#plt.title('Relação de confirmados e mortes por dia')
#plt.show()

# função .max do pandas extrai a maior data da coluna
data_maxima = dados['date'].max()

# LETRA B
# consultar todos os dados do dia data_maxima (12/05/2020), somente por estado
dados_grafico_2 = dados[(dados['date'] == data_maxima) & (dados['place_type'] == 'state')]
dados_grafico_2 = dados_grafico_2[['date', 'state', 'confirmed', 'deaths']]

# imprimir grafico de barras
#dados_grafico_2.plot.bar(x='state')
#plt.title('Casos confirmados e óbitos por COVID-19 por Estado em ')
#plt.show()

# LETRA C
# consultar todos os dados do dia data_maxima (12/05/2020), somente por estado
dados_grafico_3 = dados[(dados['date'] == data_maxima) & (dados['place_type'] == 'state')]
dados_grafico_3 = dados_grafico_2[['state', 'deaths']]
print(dados_grafico_3)