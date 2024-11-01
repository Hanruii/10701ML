import numpy as np

class NeuralNetwork:
    """
    Feedforward Neural Network with Sigmoid Activation
    F(x) = W_{L-1}' * s(W_{L-2}' * s(... s(W_0' * x + b_0)) + b_{L-2}) + b_{L-1}
    Sigmoid activation: s(x) = 1 / (1 + exp(-x))
    Note: The last layer (output layer) does not have sigmoid activation
    """
    def __init__(self):
        self.weights = []  # List of W
        self.biases = []   # List of b
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def add_layer(self, d_in, d_out):
        """
        Add a layer to the neural network
        Dimension of the layer: (d_in, d_out)
        """
        W = np.random.randn(d_in, d_out)
        self.weights.append(W)
        b = np.random.randn(d_out)
        self.biases.append(b)

    def forward(self, x):
        """
        Compute F(x)
        Input:
            x    - A numpy array of size (d_0,)
        Returns: 
            F(x) - A numpy array of size (d_L,)
        """
        # number of layers in the neural network
        L = len(self.biases) 
        #Linear forward
        #x_tilde = np.dot(self.weights.T, x)+self.biases
        #Sigmoid
        #x_t1 = 1/(1+np.exp(-x_tilde))
        x_prev = x
        for l in range(L-1):
            # Fx = self.sigmoid(self.weights[l].T @ xprev + self.biases[l])
            output = 1/(1+np.exp(-(np.dot(np.transpose(self.weights[l]), x_prev)+self.biases[l])))
            x_prev = output
    
        Fxx = np.dot(np.transpose(self.weights[L-1]), x_prev)+self.biases[L-1]
        return Fxx
       
#        raise NotImplementedError

    def backward(self, x):
        """
        Compute dF(x) / dx
        F(x) is guaranteed to be a scalar function (d_L is 1)
        Input:
            x    - A numpy array of size (d_0,)
        Returns:
            W_grad, b_grad
            W_grad  - List of gradients w.r.t. W
            b_grad  - List of gradients w.r.t. b
            Each element is a numpy array
            W_grad[i] should have the same shape as self.weights[i]
            b_grad[i] should have the same shape as self.biases[i]
        """
#        raise NotImplementedError
        # Hint: Your function should look like this
        # W_grad = []
        # b_grad = []
        # (Compute the gradients and fill in W_grad and b_grad...)
        # return W_grad, b_grad
        #ds/dm=s*(1-s) 
        """
        s = []
        s.append(x)
        x_prev = x
        for l in range(L-1):
            # Fx = self.sigmoid(self.weights[l].T @ xprev + self.biases[l])
            output = 1/(1+np.exp(-(np.dot(np.transpose(self.weights[l]), x_prev)+self.biases[l])))
            s.append(output)
            x_prev = output    
        #Fx.append(np.dot(np.transpose(self.weights[L-1]), x_prev)+self.biases[L-1])
        W_grad = []
        b_grad = []
        def sigmoid_deriv(self, x):
            ds = x * (1 - x)
            
        return ds 
        
        L = len(self.biases) 
        for k in range(L):
            ws = self.weights[-(L-1-k):]
            df = s[L-1-k]
            W_grad.append[k] = s[]      
        """
        L = len(self.biases)
        s = []
        W_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]
        x_prev = x
        for l in range(L-1):
            # Fx = self.sigmoid(self.weights[l].T @ xprev + self.biases[l])
            output = 1/(1+np.exp(-(np.dot(np.transpose(self.weights[l]), x_prev)+self.biases[l])))
            s.append(output)
            x_prev = output
        
        for k in range(L-1, -1, -1):
            inter = s[k]*(1-s[k])*(np.dot(self.weights[k+1], inter)) if k != L-1 else np.array([1])
            b_grad[k] = inter
            W_grad[k] = np.dot(np.transpose(np.mat(s[k-1])), np.mat(inter)) if k!= 0 else np.matmul(np.transpose([x]), [inter])

        return (W_grad, b_grad)

def random_weights(dims, seed=0):
    np.random.seed(seed)
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        weights.append(np.random.randn(dims[i], dims[i + 1]))
        biases.append(np.random.randn(dims[i + 1]))
    x = np.random.randn(dims[0])
    return weights, biases, x



if __name__ == '__main__':
    # Sample Input/Output
    print('Sample Input 1:')
    net = NeuralNetwork()
    net.add_layer(2, 3)
    net.add_layer(3, 1)
    net.weights = [np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), np.array([[-0.1], [-0.2], [-0.3]])]
    net.biases = [np.array([0.2, 0.0, -0.2]), np.array([0.5])]
    print(net.forward(np.array([1., 2.])))
    print(net.backward(np.array([1., 2.])))
    del net

    # The output should be
    # [0.03551854]
    # ([array([[-0.01873699, -0.03557889, -0.05048951],
    #     [-0.03747398, -0.07115778, -0.10097902]]), array([[0.75026011],
    #     [0.76852478],
    #     [0.78583498]])], [array([-0.01873699, -0.03557889, -0.05048951]), array([1.])])

    print('\nSample Input 2:')
    net = NeuralNetwork()
    net.add_layer(3, 5)
    net.add_layer(5, 5)
    net.add_layer(5, 1)
    net.weights, net.biases, x = random_weights([3, 5, 5, 1], 100)
    print(net.backward(x))

    # The output should be
    # ([array([[ 0.04851318,  0.13269722, -0.03970156, -0.08436668, -0.03228925],
    #    [-0.07030136, -0.19229402,  0.05753227,  0.12225733,  0.04679095],
    #    [ 0.00447932,  0.01225221, -0.00366573, -0.00778975, -0.00298133]]), array([[-0.26237547, -0.10951521, -0.20738717, -0.03645862,  0.01386228],
    #    [-0.10556064, -0.04406089, -0.08343738, -0.01466827,  0.00557717],
    #    [-0.06545357, -0.02732024, -0.0517359 , -0.00909516,  0.00345816],
    #    [-0.1098521 , -0.04585214, -0.08682945, -0.0152646 ,  0.0058039 ],
    #    [-0.04559855, -0.01903278, -0.03604207, -0.00633619,  0.00240914]]), array([[0.4594169 ],
    #    [0.66119191],
    #    [0.45624764],
    #    [0.05224928],
    #    [0.82681348]])], [array([-0.05626495, -0.1539005 ,  0.04604535,  0.09784737,  0.03744865]), array([-0.29504775, -0.12315259, -0.23321204, -0.04099863,  0.01558848]), array([1.])])


    print('\nSample Input 3:')
    net = NeuralNetwork()
    net.add_layer(3, 3)
    net.add_layer(3, 3)
    net.add_layer(3, 3)
    net.add_layer(3, 5)
    net.add_layer(5, 3)
    net.add_layer(3, 1)
    net.weights, net.biases, x = random_weights([3, 3, 3, 5, 3, 1], 200)
    print(net.backward(x))

    # The output should be
    # ([array([[ 0.00029623, -0.00031993, -0.00022218],
    #     [ 0.00676094, -0.00730168, -0.00507087],
    #     [-0.00048985,  0.00052902,  0.0003674 ]]), array([[-0.00935379,  0.00306617, -0.01966795],
    #     [-0.00574059,  0.00188176, -0.01207058],
    #     [-0.01000388,  0.00327927, -0.02103488]]), array([[-5.57504792e-05,  2.52067947e-02, -2.02553225e-02,
    #         6.29409573e-04, -9.23222373e-03],
    #     [-5.57501721e-05,  2.52066559e-02, -2.02552109e-02,
    #         6.29406107e-04, -9.23217289e-03],
    #     [-6.35939364e-05,  2.87531035e-02, -2.31050155e-02,
    #         7.17960329e-04, -1.05310924e-02]]), array([[-0.03591907,  0.01151784, -0.01006428],
    #     [-0.03233065,  0.01036717, -0.00905883],
    #     [-0.09458392,  0.03032936, -0.02650177],
    #     [-0.11540146,  0.03700473, -0.0323347 ],
    #     [-0.09336297,  0.02993785, -0.02615967]]), array([[0.5621692 ],
    #     [0.7436448 ],
    #     [0.87011252]])], [array([-0.00984808,  0.01063573,  0.0073863 ]), array([-0.01281307,  0.00420012, -0.02694169]), array([-9.09520237e-05,  4.11226777e-02, -3.30447844e-02,  1.02682659e-03,
    #     -1.50615644e-02]), array([-0.11885085,  0.03811081, -0.0333012 ]), array([1.])])
