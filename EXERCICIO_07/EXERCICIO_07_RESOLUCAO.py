from aula_05_neural_network import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt

arquitetura = [
	{"length_x": 2, "n_perceptrons": 1, "activation": "sigmoid"},
]

#OR
# taxa de aprendizado = 30
# épocas = 2

#AND
# taxa de aprendizado = 19
# épocas = 4

#XOR

rede = NeuralNetwork(arquitetura, 30.0, 2)

x = np.array(((0.,0.),
              (0.,1.),
              (1.,0.),
              (1.,1.)))

y_OR = np.array((0.,1.,1.,1.))
y_AND = np.array((0.,0.,0.,1.))
y_XOR = np.array((0.,1.,1.,0.))


rede.fit(x, y_OR)

print(rede.predict(x))

#plt.plot(rede.loss)
#plt.show()






