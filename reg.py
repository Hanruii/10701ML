import numpy as np

class LogisticRegression:
    def __init__(self, d):
        self.w = np.random.randn(d)

    def compute_loss(self, X, Y):
        """
        Compute l(w) with n samples.
        Inputs:
            X  - A numpy array of size (n, d). Each row is a sample.
            Y  - A numpy array of size (n,). Each element is 0 or 1.
        Returns:
            A float.
        """
        lw = np.sum(np.log(1+np.exp(np.dot(self.w, X.T)))-np.multiply(np.dot(self.w, X.T),Y))/len(Y)
        return lw
        raise NotImplementedError

    def compute_grad(self, X, Y):
        """
        Compute the derivative of l(w).
        Inputs: Same as above.
        Returns:
            A numpy array of size (d,).
        """
        dlw = np.dot((np.exp(np.dot(self.w, X.T))/(1+np.exp(np.dot(self.w, X.T)))-Y), X)/len(Y)
        return dlw
        raise NotImplementedError

    def train(self, X, Y, eta, rho):
        """
        Train the model with gradient descent.
        Update self.w with the algorithm listed in the problem.
        Returns: Nothing.
        """
        norm = 1
        while norm >= rho:
            self.w -= LogisticRegression.compute_grad(self, X, Y)*eta
            norm = np.linalg.norm(LogisticRegression.compute_grad(self, X, Y))
        #raise NotImplementedError


if __name__ == '__main__':
    # Sample Input/Output
    d = 10
    n = 1000

    np.random.seed(0)
    X = np.random.randn(n, d)
    Y = np.array([0] * (n // 2) + [1] * (n // 2))
    eta = 1e-3
    rho = 1e-6

    reg = LogisticRegression(d)
    reg.train(X, Y, eta, rho)
    print(reg.w)

    # The output should be close to
    # [ 0.15289573 -0.063752   -0.06434498 -0.02005378  0.07812127 -0.04307333
    #  -0.0691539  -0.02769485 -0.04193284 -0.01156307]
    # Error should be less than 0.001 for each element