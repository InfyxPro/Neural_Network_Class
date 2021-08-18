# Example using the neural_network class to solve a XOR gate input output
import Neural_Network, numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def map( x,  in_min,  in_max,  out_min,  out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if "__main__" == __name__:
    print("Starting...")

    errorResolution = 0.01
    """ alphaMin = 0.0001
    alphaMax = 0.1
    epsMin = 0.01
    epsMax = 2

    x = []
    y = []
    z = []
    for ialpha in range(50):
        for iepselon in range(50):
            print(ialpha, iepselon)
            z_sum = 0
            alpha = map(ialpha, 0, 75, alphaMin, alphaMax)
            eps = map(iepselon, 0, 75, epsMin, epsMax)
            network = Neural_Network.Network(2,1,1,2, eps, alpha)
            for i in range(150):
                epoch = 0
                isNotDone = True
                network.randomize_network()
                while (isNotDone):
                    isNotDone = False

                    output = network.compute_network(numpy.array([[-1], [-1] ]), True)[0][0]
                    network.calculate_gradients([-1])
                    network.back_propagation()
                    if (Neural_Network.get_global_linear_error([-1], [output]) > errorResolution): isNotDone = True
                    
                    output = network.compute_network(numpy.array([[1], [-1] ]), True)[0][0]
                    network.calculate_gradients([1])
                    network.back_propagation()
                    if (Neural_Network.get_global_linear_error([1], [output]) > errorResolution): isNotDone = True

                    output = network.compute_network(numpy.array([[-1], [1] ]), True)[0][0]
                    network.calculate_gradients([1])
                    network.back_propagation()
                    if (Neural_Network.get_global_linear_error([1], [output]) > errorResolution): isNotDone = True

                    output = network.compute_network(numpy.array([[1], [1] ]), True)[0][0]
                    network.calculate_gradients([-1])
                    network.back_propagation()
                    if (Neural_Network.get_global_linear_error([-1], [output]) > errorResolution): isNotDone = True
                    epoch += 1

                    if epoch == 500: break
                z_sum += epoch
            x.append(alpha)
            y.append(eps)
            z.append(z_sum)

    print("Done finished")

    fig = plt.figure()
 
    # syntax for 3-D plotting
    ax = plt.axes(projection ='3d')
    
    # syntax for plotting
    ax.scatter(x, y, z)
    ax.set_title('Surface plot geeks for geeks')
    plt.show() """

    network = Neural_Network.Network(2,1,1,2, 0.4, 0.001)
    network.randomize_network()
    isNotDone = True
    epoch = 0
    while (isNotDone):
        isNotDone = False
        errorr = 0

        output = network.compute_network(numpy.array([[-1], [-1] ]), True)
        network.calculate_gradients([-1])
        network.back_propagation()
        if (Neural_Network.get_global_linear_error([-1], [output]) > errorResolution): isNotDone = True
        errorr +=Neural_Network.get_global_linear_error([-1], [output])
        
        output = network.compute_network(numpy.array([[1], [-1] ]), True)
        network.calculate_gradients([1])
        network.back_propagation()
        if (Neural_Network.get_global_linear_error([1], [output]) > errorResolution): isNotDone = True
        errorr +=Neural_Network.get_global_linear_error([1], [output])

        output = network.compute_network(numpy.array([[-1], [1] ]), True)
        network.calculate_gradients([1])
        network.back_propagation()
        if (Neural_Network.get_global_linear_error([1], [output]) > errorResolution): isNotDone = True
        errorr +=Neural_Network.get_global_linear_error([1], [output])

        output = network.compute_network(numpy.array([[1], [1] ]), True)
        network.calculate_gradients([-1])
        network.back_propagation()
        if (Neural_Network.get_global_linear_error([-1], [output]) > errorResolution): isNotDone = True
        errorr +=Neural_Network.get_global_linear_error([-1], [output])
        epoch += 1
        errorr /= 4
        print(errorr)

        if epoch % 50 == 0: 
            network.randomize_network()
        elif epoch == 1000: break
                
    print("Done finished: ", epoch)
    output = network.compute_network(numpy.array([[-1], [-1] ]))[0][0]
    print("Input 0 0 Result:", output)

    output = network.compute_network(numpy.array([[1], [-1] ]))[0][0]
    print("Input 1 0 Result:", output)
    output = network.compute_network(numpy.array([[-1], [1] ]))[0][0]
    print("Input 0 1 Result:", output)
    output = network.compute_network(numpy.array([[1], [1] ]))[0][0]
    print("Input 1 1 Result:", output) 
