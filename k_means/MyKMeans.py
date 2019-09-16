import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class KMeans:
    def __init__(self, img, k=6):
        self.k = k
        self.img = img
        self.img_re = img
        self.A = []
        self.mu = []
        
        self.back_h = img.size[1]
        self.back_w = img.size[0]
        
    def img_reduction(self, factor=4, img=None):
        if img is None:
            img = self.img
            
        h = img.size[0]
        w = img.size[1]
        self.back_h = h
        self.back_w = w
        print("Original Shape : {}".format(np.array(img).shape))
        img_re = img.resize([h//factor, w//factor])
        print("Reducted Shape : {}".format(np.array(img_re).shape))
        
        self.back_h = img_re.size[1]
        self.back_w = img_re.size[0]
        self.img_re = img_re
        
        self.imshow(self.img)
        self.imshow(self.img_re)

    def img_reshape(self, x):
        x_np = np.array(x)
        return x_np.reshape(x_np.shape[0]*x_np.shape[1],x_np.shape[2])

    def img_backshape(self, x, h, w):
        return x.reshape(h, w, 3)
    
    def imshow(self, img=None, prt_shape=True):
        if img is None:
            img = self.img
            
        plt.figure(figsize=(12,12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        if prt_shape:
            print("Shape : {}".format(np.array(img).shape))
        
    def fit(self, img=None, k=None, n_iter=10):
        if img is None:
            img = self.img_re
        if k is None:
            k = self.k
        A = self.img_reshape(img)
        centroid = np.array([np.linspace(5, 255, k),
                             np.linspace(5, 255, k),
                             np.linspace(5, 255, k)]).T
        mu = centroid.copy()
        y = np.zeros([A.shape[0],1])
        d = np.zeros([k,1])
        
        for iteration in range(n_iter):
            for i in range(A.shape[0]):
                for j in range(self.k):
                    d[j] = np.linalg.norm(A[i,:]-mu[j,:], 2)
                y[i] = np.argmin(d)

            err = 0
            for i in range(self.k):
                mu[i,:] = np.mean(A[np.where(y == i)[0]], axis=0)
                err += np.linalg.norm(centroid[i,:] - mu[i,:], 2)
            centroid = mu.copy()
            print("err : {}, iter : {}/{}".format(err, iteration+1, n_iter))
            
        mu_int = mu.astype(int)
        for i in range(A.shape[0]):
            for j in range(self.k):
                d[j] = np.linalg.norm(A[i,:]-mu_int[j,:], 2)
            y[i] = np.argmin(d)
            A[i] = mu[int(y[i])]
            
        self.A = A.copy()
        self.mu = mu.copy()
    
    def show_palette(self):
        plt.figure(figsize=(12,6))
        plt.imshow(self.mu.astype(int).reshape(1,self.k,3))
        plt.axis('off')
        plt.show()
    
    def show_result(self):
        self.imshow(self.img_re, prt_shape=False)
        
        self.show_palette()
        
        A_back = self.img_backshape(self.A, self.back_h, self.back_w)
        plt.figure(figsize=(12,12))
        plt.axis('off')
        plt.imshow(A_back)
        plt.show()