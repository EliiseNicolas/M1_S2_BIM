import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def mse(W, X, Y):
    """
    Compute MSE, mean square error, cost function of linear regression

    parameters : 
        - W : np.ndarray of shape (d, 1), weights
        - X : np.ndarray of shape (n, d), data
        - Y : np.ndarray of shape (n, 1), label corresponding to data X

    returns : 
        - mse : np.ndarray of shape (n, 1), the mse 
    """
    mse = np.square((np.dot(X, W) - Y))

    return mse

def mse_grad(W, X, Y):
    """
    Compute the gradient of mse for each point of X

    parameters : 
        - W : np.ndarray of shape (d,), weights
        - X : np.ndarray of shape (n, d), data
        - Y : np.ndarray of shape (n,), label corresponding to data X

    returns :
        - grad_mse : np.ndarray of shape(n, d) the gradient of the mse for each point of X
    """
    grad_mse = 2 * X * (X.T @ W - Y) 

    return grad_mse

def reglog(W, X, Y):
    """
    Compute cost function of logistic regression : -log likelihood = 1+exp(-ywx)

    parameters : 
        - W : np.ndarray of shape (d,), weights
        - X : np.ndarray of shape (x, d), data
        - Y : np.ndarray of shape (x,), label corresponding to data X

    returns :
        - LL : np.ndarray of shape (n, 1)
    """
    # Compute loglikelihood log(P(Y|X))
    LL = np.log(1+np.exp(-Y*X@W))

    return LL

def reglog_grad(W, X, Y):
    """
    Compute gradient of cost function of logistic regression

    parameters : 
        - W : np.ndarray of shape (d,), weights
        - X : np.ndarray of shape (x, d), data
        - Y : np.ndarray of shape (x,), label corresponding to data X

    returns :
        - grad : np.ndarray of shape (n, 1)
    """
    # Compute derivative of -loglikelihood (cost function of logistic regression)
    grad = -Y*X/(1+np.exp(Y*X@W))
    return grad # a tester

def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()


if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w  = np.random.randn(datax.shape[1],1)
    plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    plot_data(datax,datay)

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
    
