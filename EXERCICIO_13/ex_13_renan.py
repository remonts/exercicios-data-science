import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importando os dados com o Pandas
dados = pd.read_csv('car-prices.csv')

# Renomeando as colunas usando rename do Pandas
dados.rename(columns = {'mileage_per_year': 'milhas_por_ano', 'model_year': 'ano_modelo',\
        'price': 'preco', 'sold': 'vendido'}, inplace = True)
# Imprimindo nome das colunas
#print(dados.columns)

# Criando nova coluna vendidos_0_1, transformando dados em 0 ou 1
dados['vendido_0_1'] = dados['vendido'].map(lambda x:\
            1 if x == 'yes' else (0 if x == 'no' else 'vazio'))

# Criando nova coluna idade_veiculo
dados['idade_modelo'] = dados['ano_modelo'].map(lambda x: 2020 - x)
d_idade_modelo = dados['idade_modelo']

# Criando nova coluna km_veiculo
dados['km_veiculo'] = dados['milhas_por_ano'].map(lambda x:\
     x * 1.60934 * d_idade_modelo)

# Excluindo as colunas pedidas no ex
dados = dados[['km_veiculo', 'idade_modelo', 'preco', 'vendido_0_1']]

# Plotando gráfico
ax = sns.scatterplot(x="km_veiculo", y="preco", data=dados)
plt.show()

# Dividindo as colunas do dataset - dados_y = coluna dos resultados
y = dados['vendido_0_1']

# Colunas x = colunas de teste
x = dados.drop('vendido_0_1', axis=1)

 # Dividindo massa de dados entre treino e teste
treino_x, teste_x, treino_y, teste_y = \
        train_test_split(x, y, random_state=None, test_size=0.25, stratify=y)

# Classificar com DummyClassifier
classificar = DummyClassifier(strategy='stratified',random_state=None, constant=None)

# Treinar com DummyClassifier
classificar.fit(treino_x, treino_y, sample_weight=None)
y_predic = classificar.predict(teste_x)

# Verificando a taxa de acerto
taxa_de_acerto = accuracy_score(y_predic, self.teste_y)
print('Taxa de acerto: ', taxa_de_acerto * 100)
