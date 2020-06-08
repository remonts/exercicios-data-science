# pandas é responsável por importar o arquivo csv
import pandas as pd
# dividir os dados de treino e testes
from sklearn.model_selection import train_test_split
# importar classe rede neural do arquivo dele
from aula_05_neural_network import NeuralNetwork
# matplotlib é para plotar graficos
import matplotlib.pyplot as plt
# numpy para trabalhar com md - matrizes multidimensionais
import numpy as np 


# importando os dados para a variavel dados / names - insere nomes nas colunas
dados = pd.read_csv("Bicicletas.csv")

# dividindo as colunas / passando a coluna Produtor para y
y = dados['bicicletas_alugadas']

# deletar a coluna produtor e passar as restantes para x / axis = eixo
x = dados.drop('bicicletas_alugadas', axis=1)

# escalar dados para ficarem na mesma escala e ser possível treinar a rede
# precisamos que os dados de entrada e saída fiquem em uma escala entre 0 e 1
x = x / np.amax(x, axis=0) # np.amax retorna o maior valor de x
x = x.values 

y_max = np.amax(y, axis=0) # np.amax retorna o maior valor de x
y = y / y_max
y = y.values

# dividindo os dados de treino e teste
# random_state - gerador de numeros aleatorios
# stratify - garante os dados de treinamento e teste com as mesmas proporções
# remover stratify pois as variações de entradas são grandes
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=20, test_size=0.25)

# arquitetura da rede
# descobrir número de colunas de entrada
# print(x.shape[1])
x_lenght = x.shape[1]

arquitetura = [
	{"length_x": x_lenght, "n_perceptrons": 100, "activation": "relu"},
    {"length_x": 100, "n_perceptrons": 1, "activation": "sigmoid"}
]
# como estamos usando ativação sigmoid, as saídas da rede estarao entre 0 e 1

# criando a rede neural - com arquitetura, taxa de aprend e n de epocas
rede = NeuralNetwork(arquitetura, 0.5, 300)

# metodo fit responsável por treinar a rede
rede.fit(treino_x, treino_y)

# mostrando as previsões da rede
previsoes_y = rede.predict(teste_x)

#print(teste_y * y_max)
#print(previsoes_y * y_max)

# imprimindo os resultados como pontos no gráfico
# np.transpose é para imprimir o teste_x na mesma proporção do y
#print(np.transpose(teste_x)[0])
#plt.plot(np.transpose(teste_x)[0], teste_y * y_max, '.')

# reshape deixa o array na mesma dimensão
#plt.plot(np.transpose(teste_x)[0], previsoes_y.reshape(-1, 1) * y_max, '.r')
#plt.show()

# imprimindo a variavel loss da rede neural
plt.plot(rede.loss)
plt.show()

# imprimir o cabeçalho do dataset
#print(dados.head())
# imprimir o shape (estrutura)
#print(dados.shape)

# imprimir treino_y para verificar como ficou essa coluna
# value_counts verifica qts valores estao sendo treinados para cada produtor (1, 2 e 3)
#print(treino_y.value_counts())
#print(treino_x.shape)