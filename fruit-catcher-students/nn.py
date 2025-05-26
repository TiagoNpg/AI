import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        # hidden_architecture is a tuple with the number of neurons in each hidden layer
        # e.g. (5, 2) corresponds to a neural network with 2 hidden layers in which the first has 5 neurons and the second has 2
        self.hidden_architecture = hidden_architecture
        # The activations are functions 
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        ''' Computes the number of weights in the neural network.'''
        # perceptron model
        if not self.hidden_architecture:
            return self.input_size + 1
        
        total = 0
        prev_size = self.input_size
        
        #Calcular na camada oculta
        for hidden_size in self.hidden_architecture: # pesos vão ser = multiplicar as hiden layers pelas anteriores pois pesos é N * N
            total += (prev_size + 1) * hidden_size   #pelo número de neurônios + 1 para o bias (1 neuronio de output)
            prev_size = hidden_size  # Atualiza o tamanho da camada anterior para a próxima iteração
        
        #Calcular na camada de output
        total += prev_size * 1 + 1   # ultima camada é a hidden layer * Neuronios de output que é 1 + 1 para o bias
        return total
        
    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w+1:]
    
    
    def forward(self, x):
        # Forward pass through the network
        
        x = np.array(x)
        
        for weights, biases in zip(self.hidden_weights, self.hidden_biases): # calcular as camadas ocultas sum(Output(x)*w + b)
            x = np.dot(x, weights) + biases 
            x = self.hidden_activation(x) #Sigmoide do output da camada oculta
    
        x = np.dot(x, self.output_weights) + self.output_bias #Para calcular o forward no output layer -> x*w(output) + b(output)
        return self.output_activation(x)  #Devolve a função de ativação do output layer, que é uma função de passo (step function) para classificar como 1 ou -1

def create_network_architecture(input_size):

    # Replace with your configuration

    hidden_fn = lambda x: 1 / (1 + np.exp(-x)) # Sigmoid activation function
    output_fn = lambda x: 1 if x > 0 else -1 # Step function for output
    return NeuralNetwork(input_size, (), hidden_fn, output_fn)