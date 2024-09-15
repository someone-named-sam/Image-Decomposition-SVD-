import numpy as np
import matplotlib.pyplot as plt
# import time
import sklearn.metrics as met
from math import log10, sqrt

photo_data = plt.imread("dog.jpg") 
plt.figure(figsize = (10,10))
plt.imshow(photo_data)
plt.title("Original Image")
plt.axis('off')
plt.show()

G = np.mean(photo_data, -1)
img = plt.imshow(G)
img.set_cmap('gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# Compute G @ G.T
G_GT = G @ G.T
# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(G_GT)
# Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
# Compute the singular values and invert them
singular_values = np.sqrt(eigenvalues)
inv_singular_values = 1.0 / singular_values
# Compute the U and Vt matrices
U = eigenvectors
S = singular_values
S_G = np.diag((S))
V = G.T @ eigenvectors @ np.diag(inv_singular_values)
Vt = V.T

# U: Left singular vectors
# S: Singular values
# Vt: Right singular vectors (transposed)

ranks = np.arange(1,min(G.shape[0],G.shape[1])+1)
mse_values = []
psnr_values=[]

for i in range(np.shape(ranks)[0]):
    
    r = ranks[i]
    compressed_U = U[:, :r]
    compressed_Vt = Vt[:r, :]
    compressed_S = S_G[0:r, :r]
    
    # G = U_G @ S_G @ V_G
    compressed_image = np.dot(np.dot(compressed_U, compressed_S), compressed_Vt)
    
    mse_value = met.mean_squared_error(compressed_image, G)
    
    mse_values.append(mse_value)
    psnr = 20 * log10(len(ranks)/ sqrt(mse_value)) 
    psnr_values.append(psnr)
    
plt.subplot(1, 2, 1)
plt.title("MSE vs ranks")
plt.plot(ranks,mse_values)


plt.subplot(1, 2, 2)
plt.title("PSNR vs ranks")
plt.plot(ranks,psnr_values)

plt.show()

# Compute G.T @ G
GT_G = G.T @ G
# Perform eigenvalue decomposition
eigenvalues_GT_G, eigenvectors_GT_G = np.linalg.eigh(GT_G)
sorted_indices_GT_G = np.argsort(eigenvalues_GT_G)[::-1]
eigenvalues_GT_G = eigenvalues_GT_G[sorted_indices_GT_G]
eigenvectors_GT_G = eigenvectors_GT_G[:, sorted_indices_GT_G]
singular_values_GT_G = np.sqrt(eigenvalues_GT_G)
inv_singular_values_GT_G = 1.0 / singular_values_GT_G
U = G @ eigenvectors_GT_G @ np.diag(inv_singular_values_GT_G)
V = eigenvectors_GT_G
Vt = V.T
S = singular_values_GT_G
S_G = np.diag((S))

# U: Left singular vectors
# S: Singular values
# Vt: Right singular vectors (transposed)

ranks = np.arange(1,min(G.shape[0],G.shape[1]))
mse_values = []
psnr_values=[]

for i in range(np.shape(ranks)[0]):
    
    r = ranks[i]
    compressed_U = U[:, :r]
    compressed_Vt = Vt[:r, :]
    compressed_S = S_G[0:r, :r]
    
    # G = U_G @ S_G @ V_G
    compressed_image = np.dot(np.dot(compressed_U, compressed_S), compressed_Vt)
    
    mse_value = met.mean_squared_error(compressed_image, G)
    mse_values.append(mse_value)
    psnr = 20 * log10(len(ranks)/ sqrt(mse_value)) 
    psnr_values.append(psnr)




plt.subplot(1, 2, 1)
plt.title("MSE vs ranks")
plt.plot(ranks,mse_values)


plt.subplot(1, 2, 2)
plt.title("PSNR vs ranks")
plt.plot(ranks,psnr_values)

plt.show()
    
    



