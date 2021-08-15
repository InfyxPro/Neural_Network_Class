# This file will host the class for neural network
# Created by Dominik Chraca on 8-13-21 (Friday the 13th ;)


import numpy
import random

class Network():
    # Param: layers = Does not include input and output layer
    def __init__(self, inputs, outputs, layers, layerWidths):
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs
        self.layerWidths = layerWidths

        self.networkArray = []

        # create the input layer
        self.networkArray.append(Layer(self.inputs, self.layerWidths))
        # create middle layers
        for i in range(self.layers):
            self.networkArray.append(Layer(self.layerWidths, self.layerWidths))
        # create output layer
        self.networkArray.append(Layer(self.layerWidths, self.outputs))

    def __str__(self):
        for i in range(len(self.networkArray)):
            print("Index: ", i)
            print("weightMatrix:\n", self.networkArray[i].weightMatrix)
            print("biasMatrix:\n", self.networkArray[i].biasMatrix)
        return "Done"

    def randomize_network(self):
        for i in range(len(self.networkArray)):
            self.networkArray[i].randomize_weightMatrix_and_biasMatrix()

    def compute_network(self, inputArray):
        for i in range(len(self.networkArray)):
            inputArray = self.networkArray[i].compute(inputArray)
        return inputArray

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
        self.__init__(self.inputs, self.outputs, self.layers, self.layerWidths)
        
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

class Layer():
    # Each layer will host the previous weight and bias matrix
    # If index == 0, then there will be no weight or bias matrix
    def __init__(self, inputs, outputs):
        self.outputs = outputs
        self.inputs = inputs

        self.weightMatrix = numpy.zeros( (self.outputs, self.inputs) )
        self.biasMatrix = numpy.zeros ( (self.outputs, 1) )

    def compute(self, inputArray):
        if len(inputArray) != self.inputs:
            raise "inputArray and weightArray have mismatch sizes"
        inputArray = self.weightMatrix.dot(inputArray)
        inputArray = inputArray + self.biasMatrix
        inputArray = self.activation_function(inputArray)
        return inputArray


    def activation_function(self, inputArray, activationFunctionInput = None):
        if (activationFunctionInput != None):
            activationFunctionInput(inputArray)
        inputArray = numpy.tanh(inputArray)

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

    oneLayer = Layer(2,4)
    oneLayer.randomize_weightMatrix_and_biasMatrix()
    print(oneLayer.compute( numpy.array([[0.5], [0] ])))

    network = Network(2, 4, 2, 5)
    print(network)
    network.randomize_network()
    print(network)
    print(network.compute_network( numpy.array([[0.5], [0] ])))

    filePath = "C:/Users/chdom/Desktop/Python-Files/normal_python/neural network"

    fullPath =  os.path.join(filePath, "Network1.txt")

    print("Save and load network test*******")
    network.save_network(fullPath)
    network.randomize_network()
    print(network)
    network.load_network(fullPath)
    print(network)

    print("Done!")
