from sklearn.datasets import load_iris
from sklearn.decomposition import PCA,NMF
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


import numpy as np

def load_data():
    X=load_iris()
    x_data,y_data=np.array(X.get('data')),np.array(X.get('target'))
    return x_data,y_data

def PCA_(x_data):
    reducation_x=PCA(n_components=2).fit_transform(x_data)
    return reducation_x
def NMF_(x_data):
    reducation_x=NMF(n_components=2).fit_transform(x_data)
    return reducation_x
def plt_class(x,y,color1,color2,color3):
    plt.scatter(x,y,c=color1)
    plt.scatter(x, y, c=color2)
    plt.scatter(x, y, c=color3)
def clt_Kmeans(x_data):
    clt=KMeans(n_clusters=3)
    clt.fit(x_data)
    return clt

def main():
    load_data()
    x_data,y_data=load_data()
    Pca_x=PCA_(x_data)
    Nmf_x=NMF_(x_data)
    clt_Pca=clt_Kmeans(Pca_x)
    clt_Nmf=clt_Kmeans(Nmf_x)
    print(clt_Pca.predict(Pca_x))

    plt.show()





if __name__=='__main__':
    main()