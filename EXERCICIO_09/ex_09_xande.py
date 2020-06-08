
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from aula_05_neural_network import NeuralNetwork


class Network():

    dados = None
    rede = None

    def __init__(self, file=None):
        self.dados = pd.read_csv(file)

        print(self.dados.shape)

        self.menu()

    def exibirDadosOriginais(self):
        print(self.dados.head())
        print(self.dados.shape)
        print(self.dados.describe())

    def exibirGrafico(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(self.dados['clima'], self.dados['temperatura'],
                   self.dados['bicicletas_alugadas'])
        plt.show()

    def treinarRede(self):
        x = self.dados.drop('bicicletas_alugadas', axis=1)
        y = self.dados['bicicletas_alugadas']

        # extraindo os maiores valores
        ymax = np.amax(y, axis=0)
        xmax = np.amax(x, axis=0)

        y = y/ymax
        x = x/xmax

        y = y.values
        x = x.values

        # dados para treino e teste
        treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=20, test_size=0.25)

        # arquitetura da rede
        arquitetura = [
            {"length_x": 2, "n_perceptrons": 100, "activation": "relu"},
            {"length_x": 100, "n_perceptrons": 1, "activation": "sigmoid"},
        ]

        self.rede = NeuralNetwork(arquitetura, 0.05, 4000)

        self.rede.fit(treino_x, treino_y)

        previsoes_y = self.rede.predict(teste_x)

        print(teste_y*ymax)
        print(previsoes_y*ymax)

        print('-----------------------------')

        plt.plot(self.rede.loss)

        plt.show()

    def menu(self):

        inputMenu = None

        while inputMenu != 4:
            os.system("cls")

            print("#### Rede Neural Para Previsão de Bicicletas alugadas ####")
            print("#### 1 - Exibir dados originais e estatísticas ####")
            print("#### 2 - Imprimir gráfico de Temperatura / Bicicletas Alugadas / Clima ####")
            print("#### 3 - Exibir Previsão da Rede Neural ####")
            inputMenu = int(input())

            if inputMenu == 1:
                print("dados Originais")
                self.exibirDadosOriginais()

            if inputMenu == 2:
                self.exibirGrafico()

            if inputMenu == 3:
                self.treinarRede()

            print("Pressione qualquer tecla...")
            input()


rede = Network("Bicicletas.csv")
