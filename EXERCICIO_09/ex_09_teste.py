import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NeuralNetwork():
    dados = None
    rede = None
    teste_y = None 
    previsoes_y = None

    def __init__(self, file=None, y=None):
        self.dados = pd.read_csv(file)

        #DIVIDINDO O DATASET ENTRE CARACTERÍSTICAS E CLASSIFICAÇÃO
        dados_x = self.dados.drop(y, axis=1)
        dados_y = self.dados[y]

        print(dados_x, dados_y)
        
        #print(self.dados.head())
		#print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
		#print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))
    

rede = NeuralNetwork('Bicicletas.csv', 'bicicletas_alugadas')