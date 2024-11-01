import numpy as np
import matplotlib.pyplot as plt


def center(x):
    x = np.array(x)
    mean = x.mean(axis=1, keepdims=True)
    return x - mean


def whitening(x):
    cov = np.cov(x)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))
    return x_whiten


def g(x):
    """The g(x) you are required to use"""
    return np.tanh(x)


def g_der(x):
    """Derivative of g(x)"""
    return 1 - g(x) * g(x)


def ica(X, iterations):
    """
    Your ICA implementation
    Input:
        X          - An (n_components, wav_length) numpy array.
        iterations - Max number of iterations to compute w_i
    Output:
        S          - An (n_components, wav_length) numpy array.
    """
    X = center(X)
    X = whitening(X)

    # Write your implementation here
    n,d = X.shape
    it = 0
    lim = 10
    M = np.zeros((n, n))
    for i in range(n):
        w_init = np.random.normal(size=(n,))
        w_init = w_init / np.linalg.norm(w_init)
        M[:, i] = w_init
        while True:
            it = it+1
            w = M[:, i]
            a = np.dot(w.T, X)
            print(a.shape)
            M[:, i] = np.mean(X*g(a))-np.mean(g_der(a))*w
            M[:, i] = M[:, i] - np.sum(np.dot(M[:, i].T, M[:, :(i-1)])*M[:, :(i-1)])
            M[:, i] = M[:, i] / np.linalg.norm(M[:, i])
            lim = np.dot(w, M[:, i])
            if lim > 1-1e-5 or it < iterations:
                S = np.dot(M.T, X)
                break
    return S
        #raise NotImplementedError


def sklearn_ica(X, iterations):
    """
    ICA implementation with sklearn
    """
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=X.shape[0], whiten='unit-variance',
                  max_iter=iterations)
    S_ = ica.fit_transform(X.T)
    return S_.T


def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    plt.subplot(3,1,3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    
    fig.tight_layout()
    plt.show()


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):
        max_val = np.max(mixtures[i])
        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5
    X = np.c_[[mix for mix in mixtures]]
    if apply_noise:
        X += 0.02 * np.random.normal(size=X.shape)
    return X


if __name__ == '__main__':
    from scipy.io import wavfile

    # Read sound files and mix
    sampling_rate, mix1 = wavfile.read('data/mix1.wav')
    sampling_rate, mix2 = wavfile.read('data/mix2.wav')
    sampling_rate, source1 = wavfile.read('data/source1.wav')
    sampling_rate, source2 = wavfile.read('data/source2.wav')
    X = mix_sources([mix1, mix2])

    # ICA
    #S = ica(X, iterations=1000)
    S = sklearn_ica(X, iterations=1000)
    plot_mixture_sources_predictions(X, [source1, source2], S)

    # Save generated sound files
    S /= (S.max() - S.min())
    wavfile.write('out1.wav', sampling_rate, S[0].astype(np.float32))
    wavfile.write('out2.wav', sampling_rate, S[1].astype(np.float32))