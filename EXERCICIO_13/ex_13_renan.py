import pandas as pd

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

dados = dados[['km_veiculo', 'idade_modelo', 'preco', 'vendido_0_1']]

print(dados.head())