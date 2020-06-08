import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NeuralNetwork():
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
	dados = None
	teste_y = None
	previsoes_y = None

	def __init__(self, file=None, y=None):
		self.dados = pd.read_csv(file, names=self.columns_name)

		#DIVIDINDO O DATASET ENTRE CARACTERÍSTICAS E CLASSIFICAÇÃO
		dados_x = self.dados.drop(y, axis=1)
		dados_y = self.dados[y]


		#DIVIDINDO MASSA DE DADOS PARA TREINAR O ALGORITMO (CARACTERÍSTICAS E CLASSIFICAÇÕES)
		#75% DOS DADOS PARA TREINAR E 25% DOS DADOS PARA TESTAR AS PREVISÕES
		SEED = 20
		np.random.seed = SEED
		treino_x, teste_x, treino_y, self.teste_y = \
			train_test_split(dados_x, dados_y, 
								test_size=0.25, stratify=dados_y)

		#CRIANDO A REDE NEURAL
		rede = MLPClassifier(hidden_layer_sizes=(13,13,13),\
							 activation='logistic', \
								max_iter=500)
		rede.fit(treino_x, treino_y)

		#ANALISANDO AS previsoes_y DA REDE PARA OS DADOS DE TESTE
		self.previsoes_y = rede.predict(teste_x)
		self.menu()

	def graph(self):
		#impressão de um gráfico de dispersão para visualização dos dados
		#cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
		ax1 = sns.scatterplot(x=self.dados['Álcool'], y=self.dados['Magnésio'], hue=self.dados['Produtor'], size=self.dados['Fenóis totais'], data=self.dados, sizes=(10, 400))
		ax1.set_title('Vinhos por Produtores')
		plt.show()

	def print_dados_originais(self):
		print(self.dados.head())
		print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
		print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))


	def print_previsao_rede(self):
		print("\n\n###DADOS PARA TESTE DA REDE###")
		print("\n\nProdutores Reais: Total: {} \n Quantidades para cada produtor: \n {} ".format(len(self.teste_y), self.teste_y.value_counts()))

		unique, counts = np.unique(self.previsoes_y, return_counts=True)
		print("\n\nPredição de Produtores pela Rede: Total: {} \n Quantidades para cada produtor: \n {} {}".format(len(self.previsoes_y), unique, counts))

		#VERIFICANDO A TAXA DE ACERTO DO ALGORITMO
		taxa_de_acerto = accuracy_score(self.teste_y, self.previsoes_y)
		print("\n\nTaxa de Acerto da Rede Neural: {}".format(taxa_de_acerto))


		print("\n\n###MATRIZ DE CONFUSÃO###")
		print(confusion_matrix(self.teste_y, self.previsoes_y))
		print(classification_report(self.teste_y, self.previsoes_y))


	def menu(self):

		while True:	
			os.system('clear')

			print("\n####REDE NEURAL PARA PREVISÃO DE PRODUTORES DE VINHO#####")
			print("\n1 - Exibir dados originais e estatísticas")
			print("2 - Imprimir gráfico de Teor Alcóolico / Magnésio / Fenóis por Produtor")
			print("3 - Exibir previsão da Rede Neural")
			opcao = int(input())

			if opcao == 1:
				self.print_dados_originais()
				input()
			if opcao == 2:
				self.graph()
			if opcao == 3:
				self.print_previsao_rede()
				input()





rede = NeuralNetwork('wine.csv', 'Produtor')






















