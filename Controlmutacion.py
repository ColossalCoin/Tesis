######### codigo 4 ########################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle


def mutacionF(sigma: float, mut_n: float, t: float, M_max: float) -> float:
    """
    Determina la probabilidad de que un individuo sufra una mutación.

    Parameters
    ----------
    sigma : float
        Un parámetro de escala que controla la magnitud de la mutación.
    mut_n : int
        DESCRIPTION.
    t : float
        DESCRIPTION.
    M_max : float
        Un número entero que representa el valor máximo posible de la probabilidad de mutación.

    Returns
    -------
    float
        la probabilidad de mutación calculada.

    """
    x0 = sigma*t/(mut_n)
    return (M_max - 5)*np.exp(-x0) + 5

if __name__=="__main__":
    with open('Case300/PoblacionInicial100_10.txt', "rb") as f:
        P_inicial = pickle.load(f)

    M_max = 20
    mut_n = 10*300
    x = np.arange(0, 100, 1)
    fig1 = plt.figure()
    plt.plot(x, mutacionF(x, mut_n, 100, M_max))
    plt.xlabel('sigma')
    plt.ylabel('% mutación')
    plt.legend()
    
    mutacionF(P_inicial.fit_std, mut_n, 1, M_max)
    mutacionF(100, P_inicial, 1000, M_max)
    mutacionF(0, P_inicial, 300, M_max)
    
    ep = np.arange(0, 1000, 10)
    
    fig2 = plt.figure()
    plt.plot(ep, mutacionF(0, mut_n, ep, M_max), label='sigma=0')
    plt.plot(ep, mutacionF(10, mut_n, ep, M_max), label='sigma=100')
    plt.plot(ep, mutacionF(20, mut_n, ep, M_max), label='sigma=200')
    plt.plot(ep, mutacionF(50, mut_n, ep, M_max), label='sigma=500')
    plt.plot(ep, mutacionF(90, mut_n, ep, M_max), label='sigma=900')
    plt.xlabel('épocas')
    plt.ylabel('% mutación')
    plt.legend()
    
    X, Y = np.meshgrid(x, ep)
    Z = mutacionF(X, mut_n, Y, M_max)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='winter')
    ax.set_xlabel('fit_std')
    ax.set_ylabel('época')
    ax.set_zlabel('% mutación')
    plt.show()
