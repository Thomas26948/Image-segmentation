import sys
sys.path.insert(0, '.')

import numpy as np
import cv2 as cv
import utils
from sklearn.cluster import KMeans


path = "C:\\Users\\Thomas\\Desktop\\MA202\\TP_Image_Segmentation-main\\TP_Image_Segmentation-main\\images\\alfa2.bmp"
cl1 = 0
cl2 = 1



m1 = 0
sig1 = 1
m2 = 1
sig2 = 1



image = cv.cvtColor(cv.imread(path),cv.COLOR_BGR2GRAY)
image = image/255


img_1d = utils.peano_transform_img(image)
Y = utils.bruit_gauss2(img_1d,cl1,cl2,m1,sig1,m2,sig2)


kmeans_clusters = 2
kmeans = KMeans(n_clusters=kmeans_clusters).fit(Y.reshape(-1,1)) #pour effectuer la segmentation
X_seg = kmeans.labels_ #variable contenant le resultat de la segmentation par le kmeans

a = Y[X_seg==0]
b = Y[X_seg==1]


proba_cl1, proba_cl2 = utils.calc_probaprio2(X_seg,cl1,cl2)
m1 = np.mean(a)
m2 = np.mean(b)
sig1 = np.std(a)
sig2 = np.std(b)



print(" estimés m1,m2,sig1,sig2 : ",m1,m2,sig1,sig2)

A = utils.calc_probatrans2(X_seg,cl1,cl2)
n = 65536

print("A : ",A)

Mat_f = utils.gauss2(Y,n,m1,sig1,m2,sig2)

print("proba cl1 : ",proba_cl1, " cl2 ",proba_cl2)

A_est, p10_est, p20_est, m1_est, sig1_est, m2_est, sig2_est = utils.estim_param_EM_mc(10, Y, A, proba_cl1, proba_cl2, m1, sig1, m2, sig2)

print("Resultat EM")
print(A_est, p10_est, p20_est)


image_segmentee = utils.MPM_chaines2(Mat_f,n,cl1,cl2,A_est,p10_est,p20_est)

print("Le taux d'erreur entre l'image segmentée et l'image réelle est de : ",utils.taux_erreur(img_1d,image_segmentee))


img_bruitee = utils.transform_peano_in_img(Y,256)
image_segmentee2D = utils.transform_peano_in_img(image_segmentee,256)


cv.imshow("Image reelle", image)
cv.imshow("Image bruitee", img_bruitee)
cv.imshow("Image segmentee", image_segmentee2D)
cv.waitKey(15000)





