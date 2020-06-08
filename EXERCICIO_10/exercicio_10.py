import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from aula_05_neural_network import NeuralNetwork


class WinePredict():
	dados = None
	clima_max = None
	temp_max = None
	ymax = None
	teste_x = None
	teste_y = None
	previsoes_y = None
	rede = None

	def __init__(self, file=None, y=None):
		columns_name = ['Produtor',
						'Álcool',
						'ácido málico',
						'cinza',
						'Alcalinidade das cinzas',
						'Magnésio',
						'Fenóis totais',
						'Flavonóides',
						'Fenóis não flavonóides',
						'Proantocianinas',
						'Intensidade da cor',
						'Matiz',
						'OD315 de vinhos diluídos',
						'Prolina']

		self.dados = pd.read_csv(file, names=columns_name)

		#DIVIDINDO O DATASET ENTRE CARACTERÍSTICAS E CLASSIFICAÇÃO
		dados_x = self.dados.drop(y, axis=1).values
		dados_y = self.dados[y].values

		x_length = dados_x.shape[1]

		#ESCALANDO OS DADOS
		dados_x = dados_x/np.amax(dados_x, axis=0)
		self.ymax = np.amax(dados_y)
		dados_y = dados_y/self.ymax

		#DIVIDINDO MASSA DE DADOS PARA TREINAR O ALGORITMO (CARACTERÍSTICAS E CLASSIFICAÇÕES)
		#75% DOS DADOS PARA TREINAR E 25% DOS DADOS PARA TESTAR AS PREVISÕES
		treino_x, self.teste_x, treino_y, self.teste_y = \
			train_test_split(dados_x, dados_y, random_state=20, test_size=0.25)
		###REDE NEURAL###

		arquitetura = [
		    {"length_x": x_length, "n_perceptrons": 100, "activation": "relu"},
		    {"length_x": 100, "n_perceptrons": 1, "activation": "sigmoid"},    
		]

		self.rede = NeuralNetwork(arquitetura, 0.05, 4000)

		self.rede.fit(treino_x, treino_y)

		#ANALISANDO AS previsoes_y DA REDE PARA OS DADOS DE TESTE
		self.previsoes_y = self.rede.predict(self.teste_x)
		self.menu()

	def graph_loss(self):
		#impressão de um gráfico de dispersão para visualização dos dados
		plt.plot(self.rede.loss)
		plt.show()

	def print_dados_originais(self):
		print(self.dados.head())
		print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
		print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))


	def print_previsao_rede(self):
		print("\n\n###DADOS PARA TESTE DA REDE###")
		print("\n\nDados de Teste: Total: {} \n".format(self.teste_y.shape[0]))

		# CHECAGEM DE ERROS POR VALOR
		# error = np.mean(teste_y != previsao_y.reshape([-1, 1]))

		x = np.transpose(self.teste_x)[0]
		y_real = self.teste_y*self.ymax
		y_predito = self.previsoes_y.reshape([-1, 1])*self.ymax

		plt.plot(x, y_real, '.', label='Dados Reais')
		plt.plot(x, y_predito, '.r', label='Previsão da Rede')
		plt.legend()
		#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
		plt.title('Previsão da Rede Neural')
		plt.xlabel('Característica 1')
		plt.ylabel('Produtores')
		plt.show()


	def menu(self):

		while True:	
			os.system('clear')

			print("\n####REDE NEURAL PARA PREVISÃO DE PRODUTORES DE VINHOS#####")
			print("\n1 - Exibir dados originais e estatísticas")
			print("2 - Exibir Taxa de Acerto da Rede Neural")
			print("3 - Exibir gráfico de Loss")
			opcao = int(input())

			if opcao == 1:
				self.print_dados_originais()
				input()
			if opcao == 2:
				self.print_previsao_rede()
			if opcao == 3:
				self.graph_loss()


bike = WinePredict('wine.csv', 'Produtor')





















