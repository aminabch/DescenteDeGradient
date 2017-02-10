# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt # pour plot functons
#plt.switch_backend('tkAgg')  # necessary for OS SUSE 13.1 version, 
# otherwise, the plt.show() function will not display any window
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
# Parametres du probleme
N=10
h=float(10**(-4))
inf =0
sup =1        # intervalle en x=[0,1]
dx = 1/ float(N+1)      # dx = pas d'espace
#Définition des fonctions que l'on va utiliser dans la suite

# Fonction pour obtenir la périodicité de la solution exacte


# Donnee de la fonctin K quon utilise ensuite
def K(x):
    return np.sqrt(x+1)
    
# Construction du vecteur de discrétisation en espace
x = np.linspace(0,1,N+2) 

# creer F

 # creer les trois vecteurs tri-diagonaux de la matrice

       
a = np.zeros(N)
for i in range (N):
       a[i] = -K((x[i]+x[i+1])/2)/(dx*dx)
       
b = np.zeros(N)
for i in range (N):
       b[i] =( K((x[i]+x[i+1])/2)+K((x[i]+x[i-1])/2))/(dx*dx)


       
#print("a: ",a)       
 # construction de la matrice tridigonale
A=sp.sparse.diags([a,b,a], [-1, 0, 1], shape=(N,N))
A=A.tocsr()
#print(A)


#u = spsolve(A,F)

#print(u)

# fonction carre utile pour apres

def u(a):
    return spsolve(A,F(a))
    
def carre(x):
    return x**2
    

#
def F(a):
    Fr = np.zeros(N)
    for k in range (N):
       Fr[k] = f(a,x[k]) # donnee de F
    return Fr
    

   
   

def v(a):
    return spsolve(A,G(a))
    

print "PRINT u ",x[9]
    
#definition de R et son gradient pour l algo de descente 
def R(a):
    return 0.5*carre((K(x[0]+x[1])*(u(a)[1]))/(dx))+0.5*carre((K(x[N]+x[N+1])*(u(a)[N-1]))/(dx))
    
def gradR(a):
    return carre(K((x[0]+x[1])*0.5))*v(a)[1]*u(a)[1]/carre(dx)+carre(K((x[N]+x[N+1])*0.5))*v(a)[N-1]*u(a)[N-1]/carre(dx)    
 
#definir p fonction negative
 
def p(x):
    return -x
    
#definir f indice a  
    
def f(a,x): 
    return p(x-a)


#definition  de AV(a) = G(a) et faire sortir V(a) 
G=np.linspace(0,0,N)

def diff(a,x):
    return float (f(a+h, x) - f(a,x)) / float(h)



def G(a):
    Gr=np.linspace(0,0,N)
    for k in range (N):
        Gr[k]= (f(a+h, x[k]) - f(a,x[k])) / float(h)
    return Gr


    
# FIN DE definiri de AV(a) = G(a) et faire sortir V(a) 

#Gradient a pas fixe pour minimisation de R(a)
rho = 0.4 #PAS CONSTANT
eps = 0.0001
w = 0
a = [0.5]

resultat = 0  #le resultat de la descente 
#
i=0
#
while np.abs(gradR(a[i]))>eps:
    w= -gradR(a[i])
    a.append(a[i]+rho*w)
   # print a[i]
    i=i+1
        

resultat = a[-1]
print resultat   #donne la valeur approchee de a qui est le minimum 


y=np.linspace(0.26, 0.74, 100)
z=np.zeros(100)

for i in range(100):
   z[i]=R(y[i])
   
plt.plot(y,z, 'bs')
plt.show()

