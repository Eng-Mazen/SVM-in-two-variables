import sklearn.svm as SVM
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
# The used functions:-
def scattering(pos, neg, size, color_pos, color_neg):
    plt.scatter(x=pos[:, 0], y=pos[:, 1], s=size, c=color_pos, marker='+', label='Positive values')
    plt.scatter(x=neg[:, 0], y=neg[:, 1], s=size, c=color_neg, marker='o', label='Negative values')
    plt.title("Positive values VS Negative values")
    plt.legend()
def Support_Vector_Machine(X, y, C=1, G=6, K='rbf'):
    model = SVM.SVC(C=C, gamma=G, kernel=K).fit(X, y)
    result = model.predict(X)
    return result ,model
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
print('''This programme works for three-file types:
      Text, Excel and MATlab files.''')
# First: loading the data file
extension = input("Enter the extension of the file (Don't forget the '.' in the start of it):\n")
# Second: Work on the conditions
# A- Excel files
if extension == '.xlsx':
    path = input("Enter the the full path after using (Ctrl+Shift+C) on the file:\n")
    data = pd.read_excel(rf"{path[1 : len(path) - 1]}")
    data = np.array(data)
    X = data[:, 0 : data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]
    condition_pos = data[:, data.shape[1] - 1] == 1
    condition_neg = data[:, data.shape[1] - 1] == 0
    positive = data[condition_pos]
    negative = data[condition_neg]
    result_of_prediction, SVC = Support_Vector_Machine(X, y)
    scattering(positive, negative, color_neg='r', color_pos='g', size=50)
    plot_svc(X=X, y=y, svc=SVC)
    plt.show()
    while True:
        predict = np.zeros(2).reshape(1, 2)
        for i in range(2):
            k = float(input(f"Enter the value number {i + 1}: "))
            predict[0, i] = k
        print('The result of this values will be:', SVC.predict(predict)[0])
        print('Want to predict in the current file again?:')
        choice = input()
        if 'y' in choice:
            continue
        else:
            print('Program has ended')
            break
# B- Text files
elif extension == '.txt':
    path = input("Enter the the full path after using (Ctrl+Shift+C) on the file:\n")
    data = pd.read_csv(rf"{path[1 : len(path) - 1]}")
    data = np.array(data)
    X = data[:, 0 : data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]
    condition_pos = data[:, data.shape[1] - 1] == 1
    condition_neg = data[:, data.shape[1] - 1] == 0
    positive = data[condition_pos]
    negative = data[condition_neg]
    result_of_prediction, SVC = Support_Vector_Machine(X, y, K='linear')
    scattering(positive, negative, color_neg='r', color_pos='g', size=50)
    plot_svc(X=X, y=y, svc=SVC)
    plt.show()
    while True:
        predict = np.zeros(2).reshape(1, 2)
        for i in range(2):
            k = float(input(f"Enter the value number {i + 1}: "))
            predict[0, i] = k
        print('The result of this values will be:', SVC.predict(predict)[0])
        print('Want to predict in the current file again?:')
        choice = input()
        if 'y' in choice:
            continue
        else:
            print('Program has ended')
            break    
# C- MATLAB files
elif extension == '.mat':
    path = input("Enter the the full path after using (Ctrl+Shift+C) on the file:\n")
    data = loadmat(rf"{path[1 : len(path) - 1]}")
    X = np.array(data['X'])
    y = np.array(data['y'])
    data = np.hstack((X, y))
    condition_pos = data[:, data.shape[1] - 1] == 1
    condition_neg = data[:, data.shape[1] - 1] == 0
    positive = data[condition_pos]
    negative = data[condition_neg]
    result_of_prediction, SVC = Support_Vector_Machine(X, y, K='linear')
    scattering(positive, negative, color_neg='r', color_pos='g', size=50)
    plot_svc(X=X, y=y, svc=SVC)
    plt.show()
    while True:
        predict = np.zeros(2).reshape(1, 2)
        for i in range(2):
            k = float(input(f"Enter the value number {i + 1}: "))
            predict[0, i] = k
        print('The result of this values will be:', SVC.predict(predict)[0])
        print('Want to predict in the current file again?:')
        choice = input()
        if 'y' in choice:
            continue
        else:
            print('Program has ended')
            break
else:
    print("Sorry, You have entered a wrong extension")