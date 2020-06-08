# -*- coding: utf-8 -*-
import numpy as np
import pprint

class NeuralNetwork():
    arquitetura = []
    weights_bias = {}
    # memoria temporaria para a retropropagacao
    memoria = {}    
    gradientes = {}    
    learning_rate = 0.01
    epochs = 1
    loss = []
    
    def __init__(self, arquitetura, learning_rate, epochs):
        self.arquitetura = arquitetura
        self.init_weights_bias()
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    
    """### Funções de ativação"""
    def activation_linear_limiar(self, vetor_z):
        return np.where(vetor_z < 0, 0, 1)

    def activation_sigmoid(self, vetor_z):
        return 1/(1+np.exp(-vetor_z))

    def activation_relu(self, vetor_z):
        return np.maximum(0,vetor_z)


    """### Derivadas das Funções de ativação - Na Retropropagação"""    
    def activation_sigmoid_back(self, derivadas_da_funcao_custo_y, saida_z_camada):
        sig = self.activation_sigmoid(saida_z_camada)
        return derivadas_da_funcao_custo_y * sig * (1 - sig)

    def activation_relu_back(self, derivadas_da_funcao_custo_y, saida_z_camada):
        dSaida = np.array(derivadas_da_funcao_custo_y, copy = True)
        dSaida[saida_z_camada <= 0] = 0;
        return dSaida;    
    

    """ Inicialização do vetor de pesos(weights) e bias da rede de acordo 
        com a arquitetura da rede.
        gera um valor aleatório para cada peso e bias
    """
    def init_weights_bias(self, seed=99):
        # inicia os valores aleatórios
        np.random.seed(seed)
        # itera nas camadas da rede
        for indice, camada in enumerate(self.arquitetura):
            indice_camada = indice + 1

            # extrai o numero de nodos nas camadas
            length_x = camada["length_x"]
            n_perceptrons = camada["n_perceptrons"]

            # inicia os valores na matriz de pesos P
            # e o vetor de viés ou bias b
            # multiplica a quantidade de entradas pela quantidade de perceptron 
            # pra saber a quantidade de pesos WEIGHTS a ser criada
            self.weights_bias['W' + str(indice_camada)] = np.random.randn(
                n_perceptrons, length_x)  * 0.1
            # a quantidade de bias é a quantidade de perceptrons    
            self.weights_bias['b' + str(indice_camada)] = np.random.randn(
                n_perceptrons, 1) * 0.1


    """
      Método que propaga uma camada por vez
    """
    def forward_one_layer(self, indice_camada, entradas_camada):
        # cálculo da entrada para a função de ativação

        #MULTIPLICACAO DE MATRIZES
        # DEFINIÇÃO:
        # NÚMERO DE COLUNAS DA MATRIZ A DEVE SER IGUAL AO NÚMERO DE LINHAS DA MATRIZ B    
        vetor_z = np.dot(
                    self.weights_bias['W' + str(indice_camada + 1)], # matriz de pesos da camada
                    entradas_camada                                  # entradas da camada
                  ) + self.weights_bias['b' + str(indice_camada + 1)] # soma com os bias da camada

        if self.arquitetura[indice_camada]['activation'] == "limiar":
            vetor_y = self.activation_linear_limiar(vetor_z)
        elif self.arquitetura[indice_camada]['activation'] == "sigmoid":
            vetor_y = self.activation_sigmoid(vetor_z)
        elif self.arquitetura[indice_camada]['activation'] == "relu":
            vetor_y = self.activation_relu(vetor_z)

        return vetor_y, vetor_z
        
        
    """
        Método que propaga a rede toda com os valores de pesos atuais
    """
    def forward_all(self, x):
        self.memoria = {}
        vetor_y = x
        for indice_camada in range(len(self.arquitetura)):
            entradas_camada = vetor_y
            vetor_y, vetor_z = self.forward_one_layer(indice_camada, entradas_camada)

            # salva as entradas da referida camada na memória - para backpropagation
            self.memoria["X" + str(indice_camada)] = entradas_camada
            # salva as saídas Z da referida camada na memória - para backpropagation
            self.memoria["Z" + str(indice_camada+1)] = vetor_z
        
        return vetor_y
        
    """
        Função de Custo para checar a o quanto a rede acertou no 
        treino - Entropia binária cruzada
        Recebe, Y predito e Y real
    """
    def get_loss_cross_entropy(self, vetor_y, real_y):
        len_vetor_y = vetor_y.shape[1]
        
        log_vetor_y_T = np.log(vetor_y).T
        dot_real_y_log_vetor_y = np.dot(real_y, log_vetor_y_T)
        
        log_vetor_y_inv_T = np.log(1 - vetor_y).T
        real_y_inv = 1 - real_y
        dot_real_y_inv_log_vetor_y_inv = np.dot(real_y_inv, log_vetor_y_inv_T)
        
        loss = -1 / len_vetor_y * \
                        (dot_real_y_log_vetor_y + dot_real_y_inv_log_vetor_y_inv)
        
        #retorna valor escalar invés de vetor
        return np.squeeze(loss)


    """
        Método que retropropaga uma camada por vez
        Entenda que a retropropagação calcula as derivadas de tudo que foi
        calculado até agora...
        Saida Y - vetor_y
        Derivadas das ativações (para a entrada original Z)
        Derivadas matriz de PESOS desta camada:
          (Y_derivada_saida * entradas originais X da camada) qtd de amostras de entrada
        Derivadas do BIAS da camada
        Derivadas da possível saída Y da camada em relação à derivada da saída atual da camada
          (Pesos Reais da camada * derivada da saída atual da camada)
          
        Regra da Cadeia
    """
    def backpropagation_one_layer(self, indice_camada, derivadas_da_funcao_custo_y):
        saida_z_camada = self.memoria["Z" + str(indice_camada+1)]        
        
        # derivada da função de ativação
        derivada_ativacao_camada = None
        if self.arquitetura[indice_camada]['activation'] == "sigmoid":
            derivada_ativacao_camada = self.activation_sigmoid_back(derivadas_da_funcao_custo_y, saida_z_camada)
        elif self.arquitetura[indice_camada]['activation'] == "relu":
            derivada_ativacao_camada = self.activation_relu_back(derivadas_da_funcao_custo_y, saida_z_camada)        

        #Resgatando as entradas reais da camada
        entradas_forward_camada = self.memoria["X" + str(indice_camada)]
        qtd_amostras_entrada_camada = entradas_forward_camada.shape[1] # [[0,0]] -> 1 [[0,0], [0, 1]] -> 2

        # derivada da matriz de Pesos desta camada
        # (derivada Y ativado * entradas reais) / qtd de amostras da camada
        
        derivadas_gradientes_pesos_esta_camada = \
            np.dot(derivada_ativacao_camada, entradas_forward_camada.T) / qtd_amostras_entrada_camada
        
        # derivada do vetor bias
        # Somatório do vetor de derivadas Z / qtds de amostras
        derivadas_gradientes_bias_esta_camada = np.sum(derivada_ativacao_camada, axis=1, keepdims=True) / qtd_amostras_entrada_camada        

        self.gradientes["dW" + str(indice_camada+1)] = derivadas_gradientes_pesos_esta_camada
        self.gradientes["db" + str(indice_camada+1)] = derivadas_gradientes_bias_esta_camada
        
        # calculando as derivadas das saídas Y para a próxima camada a retropopagar para entrar nessa função novamente
        derivadas_da_funcao_custo_y = np.dot(self.weights_bias["W" + str(indice_camada+1)].T, derivada_ativacao_camada)

        return derivadas_da_funcao_custo_y


    """
        Método que retropropaga toda a rede e retorna o 
        vetor de derivadas para as diferenças dos custos,
        também chamado de VETOR DE GRADIENTES
    """
    def backpropagation_all(self, vetor_y, real_y):
        self.gradientes = {}
        #garantindo que os 2 vetores tenham a mesma dimensão
        real_y = real_y.reshape(vetor_y.shape)        
        
        #início do cálculo do gradiente descendente
        #cálculo das derivadas do custo anterior com o custo atual para Ys        
        derivadas_da_funcao_custo_y = -(np.divide(real_y, vetor_y) - np.divide(1-real_y, 1-vetor_y))
        
        #reverso - da última camada para a primeira camada
        for indice_camada in list(reversed(range(len(self.arquitetura)))):
            derivadas_da_funcao_custo_y = \
                self.backpropagation_one_layer(indice_camada, derivadas_da_funcao_custo_y)


    """
        Método que atualiza os pesos e bias com base no vetor de gradientes
        construído na retropropagação
    """
    def update_weights_bias(self):
        for indice_camada in range(len(self.arquitetura)):
            self.weights_bias["W" + str(indice_camada+1)] -= (self.learning_rate * self.gradientes["dW" + str(indice_camada+1)])
            self.weights_bias["b" + str(indice_camada+1)] -= (self.learning_rate * self.gradientes["db" + str(indice_camada+1)])

    
    """
        Método que treina a rede
    """
    def fit(self, x, y):
        for i in range(self.epochs):
            #RECEBENDO AS PREDIÇÕES
            #print("\n\n\n###ÉPOCA: {}###".format(i+1))
            vetor_y = self.forward_all(np.transpose(x))
            
            self.loss.append(self.get_loss_cross_entropy(vetor_y, y))

            #print("\nVETOR Y:")
            #print(vetor_y)

            #print("\nPESOS ATUALIZADOS:\n\n")
            #pprint.pprint(self.weights_bias)

            #print("\nMEMÓRIA DE FORWARD:\n\n")
            #pprint.pprint(self.memoria)

            self.backpropagation_all(vetor_y, y)

            #print("\nGRADIENTES:\n\n")
            #pprint.pprint(self.gradientes)
            
            self.update_weights_bias()


    """
        Método que realiza uma predição
    """
    def predict(self, x):
        return self.forward_all(np.transpose(x))
    




