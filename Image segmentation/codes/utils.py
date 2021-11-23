import numpy as np
from math import log2, sqrt
from scipy.stats import norm


def get_line_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée selon un parcours ligne par ligne
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction ne fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours ligne par ligne
    """
    return [a.flatten() for a in np.indices((dSize, dSize))]


def line_transform_img(img):
    """
    Cette fonction prend une image carrée en entrée, et retourne l'image applatie (1 dimension) selon le parcours ligne par ligne
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0] == img.shape[1], 'veuillez donner une image carrée en entrée'
    idx = get_line_index(img.shape[0])
    return img[idx[0], idx[1]]


def transform_line_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours ligne par ligne
    :param img: un signal 1D
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize == int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_line_index(dSize)
    img = np.zeros((dSize, dSize))
    img[idx[0], idx[1]] = signal
    return img


def get_peano_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée (dont la dimension est une puissance de 2)
    selon la courbe de Hilbert-Peano
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours de Hilbert-Peano
    """
    assert log2(dSize).is_integer(), 'veuillez donne une dimension étant une puissance de 2'
    xTmp = 0
    yTmp = 0
    dirTmp = 0
    dirLookup = np.array(
        [[3, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3], [2, 3, 3, 0], [1, 0, 0, 3], [2, 1, 1, 0], [3, 2, 2, 1],
         [0, 3, 3, 2]]).T
    dirLookup = dirLookup + np.array(
        [[4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 0], [0, 4, 4, 0], [0, 4, 4, 0],
         [0, 4, 4, 0]]).T
    orderLookup = np.array(
        [[0, 2, 3, 1], [1, 0, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 3, 2, 0], [3, 2, 0, 1], [2, 0, 1, 3],
         [0, 1, 3, 2]]).T
    offsetLookup = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    for i in range(int(log2(dSize))):
        xTmp = np.array([(xTmp - 1) * 2 + offsetLookup[0, orderLookup[0, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[1, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[2, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[3, dirTmp]] + 1])

        yTmp = np.array([(yTmp - 1) * 2 + offsetLookup[1, orderLookup[0, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[1, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[2, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[3, dirTmp]] + 1])

        dirTmp = np.array([dirLookup[0, dirTmp], dirLookup[1, dirTmp], dirLookup[2, dirTmp], dirLookup[3, dirTmp]])

        xTmp = xTmp.T.flatten()
        yTmp = yTmp.T.flatten()
        dirTmp = dirTmp.flatten()

    x = - xTmp
    y = - yTmp
    return x, y


def peano_transform_img(img):
    """
    Cette fonction prend une image carrée (dont la dimension est une puissance de 2) en entrée,
    et retourne l'image applatie (1 dimension) selon le parcours de Hilbert-Peano
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0] == img.shape[1], 'veuillez donner une image carrée en entrée'
    assert log2(img.shape[0]).is_integer(), 'veuillez donne rune image dont la dimension est une puissance de 2'
    idx = get_peano_index(img.shape[0])
    return img[idx[0], idx[1]]


def transform_peano_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours de Hilbert-Peano
    :param img: un signal 1D
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize == int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_peano_index(dSize)
    img = np.zeros((dSize, dSize))
    img[idx[0], idx[1]] = signal
    return img


def MPM_gm(Y,cl1,cl2,p1,p2,m1,sig1,m2,sig2):
    """
    Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param Y: tableau des observations bruitées
    :param cl1: Valeur de la classe 1
    :param cl2: Valeur de la classe 2
    :param p1: probabilité d'apparition a priori pour la classe 1
    :param p2: probabilité d'apparition a priori pour la classe 2
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    return np.where((p1*norm.pdf(Y, m1, sig1)) > (p2*norm.pdf(Y, m2,sig2)), cl1, cl2)


def calc_param_EM_gm(Y, p1, p2, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param Y: tableau des observations bruitées
    :param p1: probabilité d'apparition a priori pour la classe 1
    :param p2: probabilité d'apparition a priori pour la classe 2
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p1, p2, m1, sig1, m2, sig2
    """

    calc_apost1 = p1*norm.pdf(Y, m1, sig1)
    calc_apost2 = p2*norm.pdf(Y, m2, sig2)
    proba_apost1 = calc_apost1 / (calc_apost1 + calc_apost2)
    proba_apost2 = calc_apost2 / (calc_apost1 + calc_apost2)
    p1 = proba_apost1.sum() / Y.shape[0]
    p2 = proba_apost2.sum() / Y.shape[0]
    m1 = (proba_apost1 * Y).sum() / proba_apost1.sum()
    sig1 = np.sqrt((proba_apost1 * ((Y - m1) ** 2)).sum() / proba_apost1.sum())
    m2 = (proba_apost2 * Y).sum() / proba_apost2.sum()
    sig2 = np.sqrt((proba_apost2 * ((Y - m2) ** 2)).sum() / proba_apost2.sum())
    return p1, p2, m1, sig1, m2, sig2


def estim_param_EM_gm(iter, Y, p1, p2, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param Y: tableau des observations bruitées
    :param p1: valeur d'initialisation de la probabilité d'apparition a priori pour la classe 1
    :param p2: valeur d'initialisation de la probabilité d'apparition a priori pour la classe 2
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de l'écart type de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de l'écart type de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p1, p2, m1, sig1, m2, sig2
    """
    p1_est = p1
    p2_est = p2
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_gm(Y, p1_est, p2_est, m1_est, sig1_est, m2_est,
                                                                     sig2_est)
        print({'p1': p1_est,'p2': p2_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})
    return p1_est, p2_est, m1_est, sig1_est, m2_est, sig2_est


def bruit_gauss2(X,cl1,cl2,m1,sig1,m2,sig2):
    n = len(X)
    bruit = np.random.normal(m1,sig1,n)
    bruit2 = np.random.normal(m2,sig2,n)
    X_cp=np.copy(X)
    for i in range(n):
        if (X[i] == cl1):
            X_cp[i] = bruit[i]
        elif (X[i] == cl2):
            X_cp[i] = bruit2[i]
    return X_cp

def taux_erreur(A,B):
    """
    :param A: un signal
    :param B: un autre signal
    :return: Pourcentage de composantes différentes entre A et B
    """
    erreur = 0
    n = len(A)
    for i in range(n):
        if (A[i] != B[i]):
            erreur += 1
    return erreur / n

def calc_probaprio2(X,cl1,cl2):
    """
    :param X: un signal
    :param cl1: classe 1 pour ω1
    :param cl2: classe 2 pour ω2
    :return: loi du processus X a priori à partir du signal d'origine X
    """
    n = len(X)
    proba_cl1 = np.sum((X==cl1))/n
    proba_cl2 = np.sum((X == cl2))/n
    return np.array([proba_cl1,proba_cl2])

    
    


def gauss2(Y,n,m1,sig1,m2,sig2):
  

    x = ( 1 / (sig1*sqrt(2*np.pi)) ) * np.exp( -1/2 * ( (Y - m1)/sig1 )**2 )
    y = ( 1 / (sig2*sqrt(2*np.pi)) ) * np.exp( -1/2 * ( (Y - m2)/sig2 )**2 )
    return np.stack((x, y), axis=1)
    


def forward2(Mat_f,A,p10,p20):
    proba= np.array([p10,p20])
    n = Mat_f.shape[0]
    alpha = np.zeros((n, 2))
    alpha[0] = proba*Mat_f[0]
    alpha[0]= alpha[0] / (alpha[0].sum())
    for i in range(1, n) :
      alpha[i] = Mat_f[i]* (alpha[i-1] @ A)
      alpha[i] = alpha[i] / (alpha[i].sum())
    return alpha
   

def backward2(Mat_f,A,p10,p20):
    n = Mat_f.shape[0]
    proba= np.array([p10,p20])

    beta = np.zeros((n, 2))
    beta[n-1] = np.ones(2)
    for i in range(n-2,-1 ,-1) :
      beta[i] = A @ (beta[i+1] * Mat_f[i+1])
      beta[i] = beta[i] / (beta[i].sum())
    return beta


def calc_probatrans2(X,cl1,cl2):
    n = len(X)
    proba1 = np.sum(X==cl1) / n
    proba2 = np.sum(X==cl2) / n
    p = np.array([proba1, proba2])
    A = np.zeros((2,2))
    for i in range(n-1) :
      if X[i] ==cl1 and X[i+1]== cl1 :
        A[0][0]+=1
      elif X[i] ==cl1 and X[i+1] == cl2 :
        A[0][1] +=1
      elif X[i] == cl2 and X[i+1] ==cl1 :
        A[1][0] +=1
      elif X[i] == cl2 and X[i+1] == cl2 :
        A[1][1] +=1
    return A/A.sum(axis=0)[:,None]


def MPM_chaines2(Mat_f,n,cl1,cl2,A,p10,p20):

    alpha = forward2(Mat_f,A,p10,p20)
    beta = backward2(Mat_f,A,p10,p20)

    proba_aposteriori = alpha*beta


    proba_aposteriori = proba_aposteriori / proba_aposteriori.sum(axis=1)[..., None]

    res = np.array([cl1,cl2])
    return res[np.argmax(proba_aposteriori, axis=1)]



def estim_param_EM_mc(iter, Y, A, p10, p20, m1, sig1, m2, sig2):
    n = len(Y)
    p10_est = p10
    p20_est = p20
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        Mat_f = gauss2(Y,n,m1_est,sig1_est,m2_est,sig2_est)
        alpha = forward2(Mat_f,A_est,p10_est,p20_est)
        beta = backward2(Mat_f,A_est,p10_est,p20_est)

        proba_posteriori=( alpha*beta) / (alpha*beta).sum() 
        p=proba_posteriori.sum(axis=0)/proba_posteriori.shape[0]

        psi= ( alpha[:-1, :, None]* (Mat_f[1:, None, :]* beta[1:, None, :]* A_est[None,:,:]) )   
        psi=psi/ (psi.sum(axis=(1,2))[:,None, None])  
        A_est= np.transpose( np.transpose( (psi.sum(axis=0))) / (proba_posteriori[:-1:].sum(axis=0)))

        m1_est= (proba_posteriori[:,0]*Y).sum()/proba_posteriori[:,0].sum()
        m2_est= (proba_posteriori[:,1]*Y).sum()/proba_posteriori[:,1].sum()
        sig1_est= np.sqrt( (proba_posteriori[:,0]*((Y-m1_est)**2)).sum()/proba_posteriori[:,0].sum())
        sig2_est= np.sqrt( (proba_posteriori[:,1]*((Y-m2_est)**2)).sum()/proba_posteriori[:,1].sum())
  

    return A_est, p10_est, p20_est, m1_est, sig1_est, m2_est, sig2_est

