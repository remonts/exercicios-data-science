import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from aula_05_neural_network import NeuralNetwork


class BikeRentPredict():
	dados = None
	clima_max = None
	temp_max = None
	ymax = None
	teste_x = None
	teste_y = None
	previsoes_y = None
	rede = None

	def __init__(self, file=None, y=None):
		self.dados = pd.read_csv(file)

		#DIVIDINDO O DATASET ENTRE CARACTERÍSTICAS E CLASSIFICAÇÃO
		dados_x = self.dados.drop(y, axis=1).values
		dados_y = self.dados[y].values

		#ESCALANDO OS DADOS
		self.clima_max = np.amax(np.transpose(dados_x)[0])
		self.temp_max = np.amax(np.transpose(dados_x)[1])

		dados_x = dados_x/np.amax(dados_x, axis=0)
		self.ymax = np.amax(dados_y)
		dados_y = dados_y/self.ymax

		#DIVIDINDO MASSA DE DADOS PARA TREINAR O ALGORITMO (CARACTERÍSTICAS E CLASSIFICAÇÕES)
		#75% DOS DADOS PARA TREINAR E 25% DOS DADOS PARA TESTAR AS PREVISÕES
		treino_x, self.teste_x, treino_y, self.teste_y = \
			train_test_split(dados_x, dados_y, random_state=43, test_size=0.25)
		###REDE NEURAL###

		arquitetura = [
		    {"length_x": 2, "n_perceptrons": 3, "activation": "relu"},
		    {"length_x": 3, "n_perceptrons": 1, "activation": "sigmoid"},    
		]

		self.rede = NeuralNetwork(arquitetura, 0.5, 1000)

		self.rede.fit(treino_x, treino_y)

		#ANALISANDO AS previsoes_y DA REDE PARA OS DADOS DE TESTE
		self.previsoes_y = self.rede.predict(self.teste_x)
		self.menu()

	def graph(self):
		#impressão de um gráfico de dispersão para visualização dos dados
		ax1 = sns.scatterplot(x=self.dados['temperatura'], 
							  y=self.dados['bicicletas_alugadas'], 
							  hue=self.dados['clima'], 
							  data=self.dados)
		ax1.set_title('Aluguel de Bicicletas por Clima e Temperatura')
		plt.show()

	def print_dados_originais(self):
		print(self.dados.head())
		print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
		print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))


	def print_previsao_rede(self, opcao):
		print("\n\n###DADOS PARA TESTE DA REDE###")
		print("\n\nDados de Teste: Total: {} \n".format(self.teste_y.shape[0]))

		# CHECAGEM DE ERROS POR VALOR
		# error = np.mean(teste_y != previsao_y.reshape([-1, 1]))

		if opcao == 'A':
			index = 0
			mult = self.clima_max
			label = 'Clima'
		elif opcao == 'B':
			index = 1
			mult = self.temp_max
			label = 'Temperatura'
		x = np.transpose(self.teste_x)[index]*mult
		y_real = self.teste_y*self.ymax
		y_predito = self.previsoes_y.reshape([-1, 1])*self.ymax

		plt.plot(x, y_real, '.', label='Dados Reais')
		plt.plot(x, y_predito, '.r', label='Previsão da Rede')
		plt.legend()
		#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
		plt.title('Previsão da Rede Neural')
		plt.xlabel(label)
		plt.ylabel('Número de bicicletas alugadas')
		plt.show()


	def menu(self):

		while True:	
			os.system('clear')

			print("\n####REDE NEURAL PARA PREVISÃO DE ALUGUEL DE BICICLETAS POR CLIMA E TEMPERATURA#####")
			print("\n1 - Exibir dados originais e estatísticas")
			print("2 - Imprimir gráfico de Clima / Temperatura / Bicicletas Alugadas")
			print("3 - Exibir Taxa de Acerto da Rede Neural")
			print("4 - Realizar previsão avulsa")
			opcao = int(input())

			if opcao == 1:
				self.print_dados_originais()
				input()
			if opcao == 2:
				self.graph()
			if opcao == 3:
				print("\n###")
				print("A - Exibir teste por clima")
				print("B - Exibir teste por temperatura")
				opcao = str(input())
				self.print_previsao_rede(opcao)
			if opcao == 4:
				print("\n###")
				print("Entre com Clima:")
				clima = int(input())
				print("Entre com Temperatura: (0 a 1)")
				temperatura = float(input())
				clima = clima/self.clima_max
				temperatura = temperatura/self.temp_max

				entrada_x = [clima, temperatura]
				print('Número de bicicletas a alugar: {}'.format(self.rede.predict([entrada_x])*self.ymax))
				input()

bike = BikeRentPredict('Bicicletas.csv', 'bicicletas_alugadas')





















