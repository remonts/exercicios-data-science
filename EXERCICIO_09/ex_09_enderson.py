import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from aula_05_neural_network import NeuralNetwork

class Teste():
      
  dados = None
  teste_y = None
  previsoes_y = None

  def __init__(self, file = None):
    self.dados = pd.read_csv(file)
    x = self.dados.drop('bicicletas_alugadas', axis=1)
    y = self.dados['bicicletas_alugadas']

    x = x/np.amax(x,axis=0)
    x = x.values
    
    y_max = np.amax(y, axis=0) # np.amax retorna o maior valor de x
    y = y / y_max
    y = y.values

    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = 20,test_size = 0.25)

    # usar sigmoid para obter os valores de saída entre 0 e 1
    arquitetura = [
                   {"length_x": 2, "n_perceptrons": 100, "activation": "relu"},
                   {"length_x": 100, "n_perceptrons": 1, "activation":"sigmoid"}
    ]
   
    rede = NeuralNetwork (arquitetura, 0.15, 8000)
    rede.fit(treino_x, treino_y)

    previsoes_y = rede.predict(teste_x)

    print(teste_y * y_max)
    print(previsoes_y * y_max)
    
    # imprimir gráfico da loss
    # plt.plot(rede.loss)
    # plt.show()

    #self.menu()
  
  """
  def graph(self):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(self.dados['clima'],self.dados['temperatura'],self.dados['bicicletas_alugadas'] )
    plt.show()
  """

  def print_dados_originais(self):
    print(self.dados.head())
    print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
    print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))

  def menu(self):

    while True:
      os.system('clear')

      print("\n#### REDE NEURAL PARA PREVISÃO DE BICICLETA ALUGADA #####")
      print("\n1 - Exibir dados originais e estatísticas")
      print("2 - Imprimir gráfico Clima / Temperatura / Bicicleta alugada")
      print("3 - Exibir previsão da Rede Neural")
      opcao = int(input())
   
      if opcao == 1:
        self.print_dados_originais()
        input()
      if opcao == 2:
        self.graph()
      if opcao == 3:
        self.print_previsao_rede()
      else:
        break

      input()
    
rede = Teste('Bicicletas.csv')