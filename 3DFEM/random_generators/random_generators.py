##############################################################################
#                                                                            #
# This Python file is part of the 3DFEM code available at:                   #
# https://github.com/rcapillon/3DFEM                                         #
# under GNU General Public License v3.0                                      #
#                                                                            #
# Code written by Rémi Capillon                                              #
#                                                                            #
##############################################################################

import numpy as np

def matrix_SG0plus(size, dispersion_coefficient):
    mat_L_T = np.zeros((size, size))
    
    sigma = dispersion_coefficient / np.sqrt(size + 1)
    beta = 1
    scale = 1 / beta
    
    for ii in range(size):
        shape_ii = (size + 1)/(2 * dispersion_coefficient**2) + (1 - ii)/2
        gamma_rand = np.random.gamma(shape_ii, scale)
        mat_L_T[ii, ii] = sigma * np.sqrt(2 * gamma_rand)
        for jj in range(ii):
            normal_rand = sigma * np.random.randn()
            mat_L_T[ii, jj] = normal_rand
        
    random_matrix = np.dot(mat_L_T,mat_L_T.transpose())
    
    return random_matrix

def matrices_wishart(n_samples, mean_cholesky_matrix, dispersion_coefficient): 
    random_matrices = np.zeros(mean_cholesky_matrix.shape + (n_samples,))
    
    for ii in range(n_samples):
        mat_G0 = matrix_SG0plus(mean_cholesky_matrix.shape[0], dispersion_coefficient)
        random_matrix = np.dot(mean_cholesky_matrix, np.dot(mat_G0, mean_cholesky_matrix.transpose()))
        random_matrices[:, :, ii] = random_matrix
    
    return random_matrices