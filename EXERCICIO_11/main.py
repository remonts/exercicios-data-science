import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np


class Covid():
    dados = None
    rede = None

    def __init__(self, file=None):
        print('teste')
        self.dados = pd.read_csv('covid.csv')
        self.menu()

    def progressionLine(self):
        dados_grafico_1 = self.dados[self.dados['place_type'] == 'state']
        dados_grafico_1 = dados_grafico_1[['date', 'state',
                                           'last_available_confirmed', 'last_available_deaths']]

        dados_grafico_1 = dados_grafico_1.groupby(['date'])[
            'last_available_confirmed', 'last_available_deaths'].sum().reset_index().sort_values(by=['date'])

        dados_grafico_1.plot.line(x='date')
        plt.title('Casos confirmados e óbitos por COVID-19 no Brasil')
        plt.show()

    def gaphicBar(self):
        data_maxima = self.dados['date'].max()

        dados_grafico_2 = self.dados[(self.dados['date'] == data_maxima)
                                     & (self.dados['place_type'] == 'state')]
        dados_grafico_2 = dados_grafico_2[[
            'date', 'state', 'last_available_confirmed', 'last_available_deaths']]

        dados_grafico_2.plot.bar(x='state')
        plt.title('Casos confirmados e óbitos por COVID-19 por estado 13/05/2020')
        plt.show()

    def graphicWordCloudBrazil(self):
        data_maxima = self.dados['date'].max()

        dados_nuvem = self.dados[(self.dados['date'] == data_maxima) &
                                 (self.dados['place_type'] == 'state')]
        dados_nuvem = dados_nuvem[[
            'date', 'state', 'last_available_confirmed', 'last_available_deaths']]

        nuvem = {}

        for i, row in dados_nuvem.iterrows():
            nuvem[row['state']] = row['last_available_deaths']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=1000)

        wc.generate_from_frequencies(nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def graphicWordCloudBrazil100k(self):
        data_maxima = self.dados['date'].max()

        dados_nuvem = self.dados[(self.dados['date'] == data_maxima) &
                                 (self.dados['place_type'] == 'state')]
        dados_nuvem = dados_nuvem[[
            'date', 'state', 'last_available_confirmed_per_100k_inhabitants']]

        nuvem = {}

        for i, row in dados_nuvem.iterrows():
            nuvem[row['state']] = row['last_available_confirmed_per_100k_inhabitants']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=1000)

        wc.generate_from_frequencies(nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def progressionLineState(self, state):
        dados_grafico_1 = self.dados[(self.dados['place_type'] == 'state') & (
            self.dados['state'] == state)]

        dados_grafico_1 = dados_grafico_1[['date', 'state',
                                           'last_available_confirmed', 'last_available_deaths']]

        dados_grafico_1 = dados_grafico_1.groupby(['date'])[
            'last_available_confirmed', 'last_available_deaths'].sum().reset_index().sort_values(by=['date'])

        dados_grafico_1.plot.line(x='date')
        plt.title('Casos confirmados e óbitos por COVID-19 no Brasil')
        plt.show()

    def gaphicBarState(self, state):
        data_maxima = self.dados['date'].max()

        dados_grafico_2 = self.dados[(self.dados['date'] == data_maxima)
                                     & (self.dados['place_type'] == 'city')
                                     & (self.dados['state'] == state)]

        dados_grafico_2 = dados_grafico_2[[
            'date', 'state', 'city', 'last_available_confirmed', 'last_available_deaths']]

        dados_grafico_2.plot.bar(x='city')
        plt.title('Casos confirmados e óbitos por COVID-19 por estado 13/05/2020')
        plt.show()

    def graphicWordCloudState(self, state):
        data_maxima = self.dados['date'].max()

        dados_nuvem = self.dados[(self.dados['date'] == data_maxima) &
                                 (self.dados['place_type'] == 'city') &
                                 (self.dados['state'] == state)]

        dados_nuvem = dados_nuvem[[
            'date', 'state', 'city', 'last_available_confirmed', 'last_available_deaths']]

        nuvem = {}

        for i, row in dados_nuvem.iterrows():
            nuvem[row['city']] = row['last_available_deaths']

        custom_mask = np.array(Image.open('mapa_brasil.png'))

        wc = WordCloud(
            background_color="white",
            mask=custom_mask,
            contour_width=3,
            contour_color='steelblue',
            max_font_size=1000)

        wc.generate_from_frequencies(nuvem)

        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def menu(self):

        while True:

            print("\n####COVID#####")
            print("\n1 - Grafico linear do BRASIL")
            print("2 - Grafico em barras por ESTADO")
            print("3 - Grafico de nuvens do BRASIL")
            print("4 - Grafico de nuvens do BRASIL por 100k")
            print("5 - Grafico linear por ESTADO")
            print("6 - Grafico em barras por CIDADE do ESTADO")
            print("7 - Grafico de nuvens por ESTADO")
            opcao = int(input())

            if opcao == 1:
                self.progressionLine()
            if opcao == 2:
                self.gaphicBar()
            if opcao == 3:
                self.graphicWordCloudBrazil()
            if opcao == 4:
                self.graphicWordCloudBrazil100k()
            if opcao == 5:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.progressionLineState(state)
            if opcao == 6:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.gaphicBarState(state)
            if opcao == 7:
                print('\nDIGITE O ESTADO: ')
                state = input()
                self.graphicWordCloudState(state)
            if opcao == 7:
                exit()


rede = Covid('Bicicletas.csv')
# dados = pd.read_csv('covid.csv')
# dadosteste = dados[['date', 'place_type', 'last_available_confirmed']]
# print(dadosteste)
