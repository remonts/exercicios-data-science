import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np


class Coronavirus():
    dados = None
    rede = None

    def __init__(self, file=None):
        self.dados = pd.read_csv('covid19_25052020.csv')
        self.menu()

    def graficoDeLinha(self):
        dados_grafico_1 = self.dados[self.dados['place_type'] == 'state']
        dados_grafico_1 = dados_grafico_1[['date', 'state', 'confirmed', 'deaths']]

        # aplicando groupby na data
        dados_grafico_1 = dados_grafico_1.groupby(['date'])['confirmed', 'deaths'].sum().reset_index().sort_values(by=['date'])

        dados_grafico_1.plot.line(x='date')
        plt.title('Casos confirmados e óbitos por COVID-19 no Brasil')
        plt.show()

    def graficoDeBarra(self):
        data_maxima = self.dados['date'].max()

        dados_grafico_2 = self.dados[(self.dados['date'] == data_maxima) & (self.dados['place_type'] == 'state')]
        dados_grafico_2 = dados_grafico_2[['date', 'state', 'confirmed', 'deaths']]

        dados_grafico_2.plot.bar(x='state')
        plt.title('Casos confirmados e óbitos por COVID-19 por estado ontem')
        plt.show()

    def nuvemDeObitosPorEstado(self):
        data_maxima = self.dados['date'].max()

        dados_grafico_3 = self.dados[(self.dados['date'] == data_maxima) & (self.dados['place_type'] == 'state')]
        dados_grafico_3 = dados_grafico_3[['date', 'state', 'confirmed', 'deaths']]

        d_nuvem = {}

        for i, row in dados_grafico_3.iterrows():
            d_nuvem[row['state']] = row['deaths']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=2000)

        wc.generate_from_frequencies(d_nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def nuvemDeIncidenciaPorEstado(self):
        data_maxima = self.dados['date'].max()

        dados_grafico_4 = self.dados[(self.dados['date'] == data_maxima) &(self.dados['place_type'] == 'state')]
        dados_grafico_4 = dados_grafico_4[['date', 'state', 'confirmed_per_100k_inhabitants']]

        d_nuvem = {}

        for i, row in dados_grafico_4.iterrows():
            d_nuvem[row['state']] = row['confirmed_per_100k_inhabitants']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=1000)

        wc.generate_from_frequencies(d_nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def graficoDeLinhaPorEstado(self, state):
        dados_grafico_1 = self.dados[(self.dados['place_type'] == 'state') & (self.dados['state'] == state)]
        dados_grafico_1 = dados_grafico_1[['date', 'state', 'confirmed', 'deaths']]

        dados_grafico_1 = dados_grafico_1.groupby(['date'])['confirmed', 'deaths'].sum().reset_index().sort_values(by=['date'])

        dados_grafico_1.plot.line(x='date')
        plt.title('Casos confirmados e óbitos por COVID-19 por estado no Brasil')
        plt.show()

    def graficoDeBarraCidadePorEstado(self, state):
        data_maxima = self.dados['date'].max()

        dados_grafico_2 = self.dados[(self.dados['date'] == data_maxima) & (self.dados['place_type'] == 'city') & (self.dados['state'] == state) & self.dados['deaths']>0]

        dados_grafico_2 = dados_grafico_2[['date', 'state', 'city', 'confirmed', 'deaths']]

        dados_grafico_2.plot.bar(x='city')
        plt.title('Casos confirmados e óbitos por COVID-19 por cidade')
        plt.show()

    def graficoDePalavrasPorEstado(self, state):        
        data_maxima = self.dados['date'].max()
        dados_grafico_5 = self.dados[(self.dados['date'] == data_maxima) & (self.dados['place_type'] == 'city') & (self.dados['state'] == state) & self.dados['deaths']>0]

        dados_grafico_5 = dados_grafico_5[['date', 'state', 'city', 'confirmed', 'deaths']]

        d_nuvem = {}

        for i, row in dados_grafico_5.iterrows():
            d_nuvem[row['city']] = row['deaths']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=2000)

        wc.generate_from_frequencies(d_nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def menu(self):

        while True:

            print("\n# PROJETO DE DATA SCIENCE CORONA VIRUS - COVID-19 #")
            print("\n1 - Gráfico de linha da evolução de casos e mortes por data no Brasil")
            print("2 - Gráfico de barras para Número de Confirmados e Mortes por Estado na data de hoje")
            print("3 - Nuvem de palavras - maior número de óbitos por Estado")
            print("4 - Nuvem de palavras - maior incidencia por Estado")
            print("5 - Gráfico de linha por estado no Brasil")
            print("6 - Gráfico de barras para Número de Confirmados por cidade do Brasil")
            print("7 - Nuvem de palavras por cidade")
            print("\nEscolha uma opção: ")
            opcao = int(input())

            if opcao == 1:
                self.graficoDeLinha()
            if opcao == 2:
                self.graficoDeBarra()
            if opcao == 3:
                self.nuvemDeObitosPorEstado()
            if opcao == 4:
                self.nuvemDeIncidenciaPorEstado()
            if opcao == 5:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.graficoDeLinhaPorEstado(state)
            if opcao == 6:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.graficoDeBarraCidadePorEstado(state)
            if opcao == 7:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.graficoDePalavrasPorEstado(state)
            else:
                exit()

rede = Coronavirus('covid19_25052020.csv')

