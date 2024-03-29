# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs
eigenvalues, eigenvectors = np.linalg.eig(adj_mat)

# Eigen vector corresponding to 1
eigen_1=(abs(eigenvectors[:,0]))/(np.linalg.norm(eigenvectors[:,0],1))
print(eigen_1)
# most important page
print(type(eigen_1))
page=np.where(np.max(eigen_1) == eigen_1)[0][0] + 1
print(page)
# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
init_I = np.array([1,0,0,0,0,0,0,0])
print(init_I)

# Perform iterations for power method
for k in range(10):
    init_I = np.matmul(adj_mat,init_I)
    init_I = init_I/np.linalg.norm(init_I, 1)


print(init_I)

power_page = np.where(np.max(init_I) == init_I)[0][0] + 1
print(power_page)
# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I
new_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for k in range(10):
    new_init_I = np.matmul(new_adj_mat,new_init_I)
    new_init_I = new_init_I/np.linalg.norm(new_init_I, 1)


print(new_init_I)



# Code ends here


# --------------
# Alpha value
alpha = 0.85

# Code starts here

# Modified adjancency matrix
G = alpha*new_adj_mat + (1-alpha)*(np.ones(8))/(8)
print(G)

# Initialize stationary vector I

final_init_I=np.array([1,0,0,0,0,0,0,0])
# Perform iterations for power method
for k in range(1000):
    final_init_I = np.matmul(G,final_init_I)
    final_init_I = final_init_I/np.linalg.norm(final_init_I, 1)


print(final_init_I)

# Code ends here


