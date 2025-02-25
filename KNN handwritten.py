import numpy as np
from math import sqrt

from sklearn.datasets import fetch_openml, load_digits
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def euclidean_distance(row1, row2) :
    distance = 0
    for i in range(len(row1) - 1) :
        distance += (row1[i] - row2[i])**2 # On calcule la différence entre les 2 points puis on la met au carré pour avoir une valeur toujours positive (différence entre les pixel)
    return sqrt(distance)


def get_neighbors(train, test_row, num) :
    distance = list()
    data = list()
    for i in train :
        dist = euclidean_distance(test_row, i) #calcul de la distance d'un point par rapport aux point du dataset
        distance.append(dist)# On ajoute la différence entre les pixels des 2 images au tableau
        data.append(i) # on Ajoute le label correspondant dans un autre tableau en parrallèle

    distance = np.array(distance) #transformation des valeurs en array numpy
    data = np.array(data)
    index_distance = distance.argsort() #récupère les index des points par ordre de distance croissante
    data = data[index_distance] #réarrange le tableau en fonction des index récupérés

    neighbors = data[:num] # on récupère les voisins les plus proches
    return neighbors


def prediction(train, test_row, num) :
    neighbors = get_neighbors(train, test_row, num)
    classes = []
    for i in neighbors :
        classes.append(i[-1]) #On ajoute les classes trouvée par la fonction get_neighbors

    pred = max(classes, key=classes.count) # on récupère la classe qui apparait le plus souvent dans le tableau (si sur 5 valeur, 1 apparaît 4 fois, on prend 1)
    return pred


def accuracy(y_true, y_pred) :
    n_correct = 0
    for i in range(len(y_true)) :
        if y_true[i] == y_pred[i] :
            n_correct += 1
    acc = n_correct/len(y_true)
    return acc


# Récupération et préparation du JDD
mnist = fetch_openml("mnist_784")
mnist.keys()
mnist.target = mnist.target.astype(np.int8)

x = np.array(mnist.data)
y = np.array(mnist.target)
print(x.shape, y.shape)

shuffle = np.random.permutation(x.shape[0])
x = x[shuffle]
y = y[shuffle]

# Créer un graphique qui affiche un chiffre
digit = x[20]
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap=matplotlib.cm.binary)
plt.axis("off")
plt.show()

# Entrainement du modèle sur une portion de 2000 image
x_train = x[:2000]
y_train = y[:2000]
train = np.insert(x_train, 784, y_train, axis=1)
predict = prediction(train, train[1244], 4)
print("predicted :", predict)
print("actual value:", train[1244][-1])

# affichage d'un des chiffre avec un graphique
digit = train[1244][:-1]
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap=matplotlib.cm.binary)
plt.axis("off")
plt.show()

# calcul de la précision du modèle
y_pred = []
y_true = train[:, -1]
for i in range(len(train)) :
    predict = prediction(train, train[i], 4)
    y_pred.append(prediction)
acc = accuracy(y_true, y_pred)
print("accuracy:", acc)



