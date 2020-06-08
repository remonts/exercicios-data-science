import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Projeto():
    dados = None
    teste_x = None
    teste_y = None
    classificar = None

    def __init__(self, file=None):    
        # Importando os dados
        self.dados = pd.read_csv(file)

        # Usando map para criar nova coluna finished e inverter valores
        self.dados['finished'] = self.dados['unfinished'].map(lambda x:\
            1 if x == 0 else (0 if x == 1 else 2))

        # Renomeando as colunas com rename do pandas
        self.dados.rename(columns = {'unfinished': 'nao_finalizado', 'expected_hours': 'horas_estimadas',\
        'price': 'preco', 'finished': 'finalizado'}, inplace = True)
    
        # Dividindo as colunas do dataset - dados_y = coluna dos resultados
        y = self.dados['finalizado']

        # Colunas x = colunas de teste
        x = self.dados.drop('finalizado', axis=1)
        
        # Escalando os dados
        # X
        #x = x / np.amax(x, axis=0) # np.amax retorna o maior valor de x
        #x = x.values

        # Y
        #y = y / np.amax(y, axis=0) # np.amax retorna o maior valor de x
        #y = y.values

        # Dividindo massa de dados entre treino e teste
        treino_x, self.teste_x, treino_y, self.teste_y = \
                train_test_split(x, y, random_state=None, test_size=0.25, stratify=y)

        # Classificar com DummyClassifier
        self.classificar = DummyClassifier(strategy='stratified',random_state=None, constant=None)

        self.classificar.fit(treino_x, treino_y, sample_weight=None)
        #y_predic = classificar.predict(self.teste_x)
            
        
        #taxa_de_acerto = accuracy_score(y_predic, self.teste_y)
        #print('Taxa de acerto: ', taxa_de_acerto * 100)
        self.menu()

    def menu(self):
		    
        while True:		    
            os.system('clear')

            print('##### EFETUANDO PREVISÕES PARA O USUÁRIO #####')
            print("\n1 - Fazer previsão de um projeto")
            print("2 - Sair")
            print("\nOpção: ")
            opcao = int(input())

            os.system('clear')          
            if opcao == 1:
                print("Entre com horas estimadas do projeto:")		    
                horas = int(input())
                print("\nEntre com preço estimado do projeto:")		
                preco = int(input())
                entrada_x = [horas, preco]                    
                print('\nRESULTADO: {}'.format(self.classificar.predict([entrada_x])))
                input()
            else:
                exit()

projeto = Projeto('projects.csv')



"""
# Gráfico de dispersão
ax = sns.scatterplot(x="horas_estimadas", y="preco", data=dados)
#plt.show()

# Projetos finalizados e não finalizados
ax = sns.scatterplot(x="horas_estimadas", y="preco", hue="finalizado", data=dados)
#plt.show()

# Projetos finalizados e não finalizados v2
ax = sns.relplot(x="horas_estimadas", y="preco", col="finalizado", hue="finalizado",\
     data=dados)
#plt.show()
"""