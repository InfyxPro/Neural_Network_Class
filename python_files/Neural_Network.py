# This file will host the class for neural network
# Created by Dominik Chraca on 8-13-21 (Friday the 13th ;)


import numpy
import random

# Public functions          ***************
def get_global_linear_error(idealData, actualData):
        # Get output layer actual data and compare it with ideal data using MSE, returns a percentage
        # Data should be 1D list type
        MSE = 0
        for i in range(len(idealData)):
            E = idealData[i] - actualData[i]
            MSE += E * E
        MSE /= len(idealData)
        return MSE

# Private functions         ***************
def hyperbolicTan(inputArray):
        inputArray = numpy.tanh(inputArray)
        return inputArray
def hyperbolicTanDerivative(inputArray):
        inputArray = 1 - numpy.tanh(inputArray) * numpy.tanh(inputArray)
        return inputArray

# Classes                   ***************
class Network():
    # Param: layers = Does not include input and output layer
    def __init__(self, inputs, outputs, layers, layerWidths, epsilon, alpha):
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs
        self.layerWidths = layerWidths
        # These variables are used for choosing the learning rate
        self.epsilon = epsilon
        self.alpha = alpha

        # Activation function, default: hyperbolicTan is used
        self.activationFunction = hyperbolicTan
        self.activationFunctionDerivative = hyperbolicTanDerivative

        # This will hold all of the real time data of each node NOTE: Does not include input layer
        self.nodeDataArray = []
        for i in range(self.layers):
            self.nodeDataArray.append(Node_Layer(layerWidths))
        # Add output layer
        self.nodeDataArray.append(Node_Layer(outputs))

        # This will hold the last inputArray values
        self.inputArray = []

        # This will hold all of the weights and bias values
        self.networkArray = []

        # create the input layer
        self.networkArray.append(Weight_Layer(self.inputs, self.layerWidths))
        # create middle layers
        for i in range(self.layers - 1):
            self.networkArray.append(Weight_Layer(self.layerWidths, self.layerWidths))
        # create output layer
        self.networkArray.append(Weight_Layer(self.layerWidths, self.outputs))
        # Create previousNetworkArray, this is used for back_propagation learning method
        self.previousNetworkArray = self.networkArray

    def __str__(self):
        for i in range(len(self.networkArray)):
            print("Index: ", i)
            print("weightMatrix:\n", self.networkArray[i].weightMatrix)
            print("biasMatrix:\n", self.networkArray[i].biasMatrix)
        return "Done"

    def randomize_network(self):
        for i in range(len(self.networkArray)):
            self.networkArray[i].randomize_weightMatrix_and_biasMatrix()

    def compute_network(self, inputArray, gatherData = False):
        # gatherData should only be on when training
        self.inputArray = inputArray
        if gatherData:
            for layer in range(len(self.networkArray)):
                # Before activation_function
                inputArray = self.networkArray[layer].compute(inputArray)
                for node in range(len(inputArray)):
                    self.nodeDataArray[layer].nodeArray[node].sumValue = inputArray[node][0]
                # After activation_function
                inputArray = self.activationFunction(inputArray)
                for node in range(len(inputArray)):
                    self.nodeDataArray[layer].nodeArray[node].outValue = inputArray[node][0]
        else: 
            for layer in range(len(self.networkArray)):
                inputArray = self.networkArray[layer].compute(inputArray)
                inputArray = self.activationFunction(inputArray)
        return numpy.transpose(inputArray)[0]

    def save_network(self, filePath):
        file = open(str(filePath), "w+")

        # First make the network header
        mainHeaderData = ""
        mainHeaderData += str(self.inputs)
        mainHeaderData += " "
        mainHeaderData += str(self.outputs)
        mainHeaderData += " "
        mainHeaderData += str(self.layers)
        mainHeaderData += " "
        mainHeaderData += str(self.layerWidths)
        mainHeaderData += "\n"
        file.write(mainHeaderData)

        for layer in self.networkArray:
            # Make layer header
            layerHeaderData = ""
            layerHeaderData += str(layer.inputs)
            layerHeaderData += " "
            layerHeaderData += str(layer.outputs)
            layerHeaderData += "\n"
            file.write(layerHeaderData)

            # ***** insert layer data
            # insert weightData
            layerWeightData = ""
            for row in range(layer.outputs):
                for column in range(layer.inputs):
                    layerWeightData += str(layer.weightMatrix[row][column]) 
                    layerWeightData += " "
            layerWeightData += "\n"
            file.write(layerWeightData)

            # insert bias data
            layerBiasData = ""
            for bias in layer.biasMatrix:
                layerBiasData += str(bias[0])
                layerBiasData += " "
            layerBiasData += "\n"
            file.write(layerBiasData)



        file.close()

    def load_network(self, filePath):
        file = open(filePath, "r")

        # Read main header for class
        mainHeaderData = file.readline().split()
        self.inputs = int(mainHeaderData[0])
        self.outputs = int(mainHeaderData[1])
        self.layers = int(mainHeaderData[2])
        self.layerWidths = int(mainHeaderData[3])
        self.__init__(self.inputs, self.outputs, self.layers, self.layerWidths, self.epsilon, self.alpha)
        
        for layer in self.networkArray:
            # read layer header
            mainLayerData = file.readline().split()
            layer.inputs = int(mainLayerData[0])
            layer.outputs = int(mainLayerData[1])

            # read weightData
            fileIndex = 0
            mainWeightData = file.readline().split()
            for row in range(layer.outputs):
                for column in range(layer.inputs):
                    layer.weightMatrix[row][column] = float(mainWeightData[fileIndex])
                    fileIndex += 1

            # read bias data
            mainBiasData = file.readline().split()
            for Ibias in range(len(layer.biasMatrix)):
                layer.biasMatrix[Ibias] = float(mainBiasData[Ibias])


        file.close()

    def calculate_gradients(self, idealData):
        # Steps to calculate the gradients
        """     • Calculate the error, based on the ideal of the training set
                • Calculate the node delta for the output neurons
                • Calculate the node delta for the interior neurons
                • Calculate individual gradients """
        # Calculate the node delta for the output neurons
        for node in range(len(idealData)):
            ErrorArray = self.nodeDataArray[-1].nodeArray[node].outValue - idealData[node]
            self.nodeDataArray[-1].nodeArray[node].nodeDelta = -1 * ErrorArray * self.activationFunctionDerivative(self.nodeDataArray[-1].nodeArray[node].sumValue)

        # Calculate the node delta for the interior neurons using backward propagation
        for layer in range(len(self.nodeDataArray)-2, -1, -1):
            for node in range(len(self.nodeDataArray[layer].nodeArray)):
                activationFunctionDerivativeValue = self.activationFunctionDerivative(self.nodeDataArray[layer].nodeArray[node].sumValue)
                summationValue = 0
                for nextLayerNode in range(len(self.nodeDataArray[layer + 1].nodeArray)):
                    nodeDelta = self.nodeDataArray[layer + 1].nodeArray[nextLayerNode].nodeDelta
                    weightValue = self.networkArray[layer + 1].weightMatrix[nextLayerNode][node]
                    summationValue += weightValue * nodeDelta
                self.nodeDataArray[layer].nodeArray[node].nodeDelta = activationFunctionDerivativeValue * summationValue
        
        # Calculate individual gradients for all weight positions
        # first start with input layer
        for node in range(len(self.inputArray)):
            output = self.inputArray[node][0]
            for nextNode in range(len(self.nodeDataArray[0].nodeArray)):
                nodeDelta = self.nodeDataArray[0].nodeArray[nextNode].nodeDelta
                self.networkArray[0].gradientMatrix[nextNode][node] = output * nodeDelta
        # Complete the gradientMatrix for the biases
        for node in range(len(self.networkArray[0].biasMatrix)):
            self.networkArray[0].gradientBiasMatrix[node] = self.nodeDataArray[0].nodeArray[node].nodeDelta

        # complete hidden layers
        for weightLayer in range(1, len(self.networkArray)):
            for node in range(len(self.nodeDataArray[weightLayer - 1].nodeArray)):
                output = self.nodeDataArray[weightLayer - 1].nodeArray[node].outValue
                for nextNode in range(len(self.nodeDataArray[weightLayer].nodeArray)):
                    nodeDelta = self.nodeDataArray[weightLayer].nodeArray[nextNode].nodeDelta
                    self.networkArray[weightLayer].gradientMatrix[nextNode][node] = output * nodeDelta
            # Complete the gradientMatrix for the biases
            for node in range(len(self.networkArray[weightLayer].biasMatrix)):
                self.networkArray[weightLayer].gradientBiasMatrix[node] = self.nodeDataArray[weightLayer].nodeArray[node].nodeDelta

    def back_propagation(self):
        # save weights array
        self.previousNetworkArray = self.networkArray

        for layer in range(len(self.networkArray)):
            for row in range(len(self.networkArray[layer].weightMatrix)):
                for col in range(len(self.networkArray[layer].weightMatrix[row])):
                    self.networkArray[layer].weightMatrix[row][col] += \
                        self.epsilon * self.networkArray[layer].gradientMatrix[row][col] + \
                        self.alpha * self.previousNetworkArray[layer].weightMatrix[row][col]
            # Need to also do same thing to biases
            for node in range(len(self.networkArray[layer].biasMatrix)):
                self.networkArray[layer].biasMatrix[node][0] += \
                    self.epsilon * self.networkArray[layer].gradientBiasMatrix[node][0] + \
                    self.alpha * self.previousNetworkArray[layer].biasMatrix[node][0]
                

        


class Node_Layer():
    # This class is used for saving the data of the last compute operation of each node 
    # Input layer should not be recorded using this class
    def __init__(self, numNodes):
        self.nodeArray = []
        for i in range(numNodes):
            self.nodeArray.append(Node())

class Node():
    def __init__(self):
        self.sumValue = 0
        self.outValue = 0
        self.nodeDelta = 0

class Weight_Layer():
    # Each layer will host the weight and bias matrix
    def __init__(self, inputs, outputs):
        self.outputs = outputs
        self.inputs = inputs

        self.weightMatrix = numpy.zeros( (self.outputs, self.inputs) )
        self.gradientMatrix = numpy.zeros( (self.outputs, self.inputs) )
        self.biasMatrix = numpy.zeros ( (self.outputs, 1) )
        self.gradientBiasMatrix = numpy.zeros ( (self.outputs, 1) )

    def compute(self, inputArray):
        if len(inputArray) != self.inputs:
            raise "inputArray and weightArray have mismatch sizes"
        inputArray = self.weightMatrix.dot(inputArray)
        inputArray = inputArray + self.biasMatrix
        return inputArray

    def randomize_weightMatrix_and_biasMatrix(self):
        for row in range(len(self.weightMatrix)):
            for column in range(len(self.weightMatrix[row])):
                self.weightMatrix[row][column] = (random.random() * 2) - 1
        
        for i in range(len(self.biasMatrix)):
            self.biasMatrix[i] = (random.random() * 2) - 1
            


if "__main__" == __name__:
    import os
    print("\n ***** Running test for the neural network...")

    oneLayer = Weight_Layer(2,4)
    oneLayer.randomize_weightMatrix_and_biasMatrix()
    print(oneLayer.compute( numpy.array([[0.5], [0] ])))

    network = Network(2, 2, 2, 5, 0.2, 0.1)
    print(network)
    network.randomize_network()
    print(network)
    print(network.compute_network( numpy.array([[0.5], [0] ]), True))
    network.calculate_gradients([1,0.5,-1])
    network.back_propagation()

    filePath = "C:/Users/chdom/OneDrive/Desktop/git/Neural_Network_Class/python_files"

    fullPath =  os.path.join(filePath, "Network1.txt")

    print("Save and load network test*******")
    network.save_network(fullPath)
    network.randomize_network()
    print(network)
    network.load_network(fullPath)
    print(network)

    print(get_global_linear_error([-0.45, 0.24], [0.4, -0.8]))


    print("Done!")
