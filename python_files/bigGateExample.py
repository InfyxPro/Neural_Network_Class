import Neural_Network, numpy, random
import matplotlib.pyplot as plt
# gate with multiple outputs
# Y = A + B
# Z = A * B
# inputs = [A, B]
# outputs = [Y, Z]

inputs = [  
    [-.5,-.5], 
    [.7,-.5],  
    [-.5,.7],  
    [.7,.7]  
    ]

inputs1 = [  
    [-0.001,-0.001], 
    [0.001,-0.001],  
    [-0.001,0.001],  
    [0.001,0.001]  
    ]

outputs = [  
    [0,1],  
    [1,0],  
    [1,0],  
    [1,0]  
    ]

network = Neural_Network.Network(2, 2, 1, 2, 0.4, 0.0042)
network.randomize_network()
isNotDone = True
epoch = 0
errorResolution = 0.01
worksWell = 0

while (worksWell != 1000):
        isNotDone = False

        for input in range(len(inputs)):
            output = network.compute_network(numpy.array([ [inputs[input][0] + random.randint(-1000, 1000)/10000], [inputs[input][1] + random.randint(-1000, 1000)/10000] ]), True)
            network.calculate_gradients(outputs[input])
            network.back_propagation()
            if (Neural_Network.get_global_linear_error(outputs[input], output) > errorResolution): isNotDone = True
        epoch += 1
        print(Neural_Network.get_global_linear_error(outputs[input], output))

        if isNotDone == False:
            worksWell += 1
        else:
            worksWell = 0

        if epoch % 4000 == 0: 
            network.randomize_network()
            pass

        #plt.scatter(epoch, Neural_Network.get_global_linear_error(outputs[input], output))
        #plt.pause(0.00001)

        #elif epoch >= 1000: break
print("Done in:", epoch)
for input in range(len(inputs1)):
            print("\nindex:", input)
            print(inputs[input])
            output = network.compute_network(numpy.array([[inputs[input][0]], [inputs[input][1]] ]), True)
            print(outputs[input])
            print(output)








