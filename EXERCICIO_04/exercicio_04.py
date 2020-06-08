import os
import math

class NeuralNetwork():
	weights_or = [0.5, 0.5]
	bias_or = -0.25

	weights_and = [1.5, 1.5]
	bias_and = -2.25


	def __init__(self):
		self.menu()

	def activation_linear_limiar(self, z):
		if z < 0:
			return 0
		else:
			return 1	

	def activation_sigmoid(self, z):
		return (1 / (1 + math.exp(-z)))

	def perceptronOR(self, entradas):
		z = 0
		for i in range(len(entradas)):
			z += entradas[i] * self.weights_or[i]
		z += self.bias_or

		y = self.activation_linear_limiar(z)
		#y = self.activation_sigmoid(z)
		return y

	def perceptronAND(self, entradas):
		z = 0
		for i in range(len(entradas)):
			z += entradas[i] * self.weights_and[i]
		z += self.bias_and

		y = self.activation_linear_limiar(z)
		#y = self.activation_sigmoid(z)
		return y
    
	def fit(self):
		return None	

	def predict(self, opcao, teste_x):
		if opcao == 1:
			return self.perceptronOR(teste_x)
		elif opcao == 2:
			return self.perceptronAND(teste_x)



	def menu(self):
		print("1 - Rede Neural para porta OU")
		print("2 - Rede Neural para porta AND")
		opcao = int(input())
		while(True):
			os.system("clear")
			if opcao == 1:
				print("###PORTA OU###")
			elif(opcao == 2):
				print("###PORTA AND###")
			print("Primeira Entrada(0-1): ")
			x1 = int(input())
			print("Segunda Entrada(0-1): ")
			x2 = int(input())
			teste_x = [x1, x2]
			print("SAÍDA: {}".format(self.predict(opcao, teste_x)))
			input()

NeuralNetwork()		
