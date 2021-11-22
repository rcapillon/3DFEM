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

def matrix_SGplus(size, dispersion_coefficient):
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

def matrices_SEplus(n_samples, mean_cholesky_matrix, dispersion_coefficient): 
    random_matrices = np.zeros(mean_cholesky_matrix.shape + (n_samples,))
    
    for ii in range(n_samples):
        mat_G0 = matrix_SGplus(mean_cholesky_matrix.shape[0], dispersion_coefficient)
        random_matrices[:, :, ii] = np.dot(mean_cholesky_matrix, np.dot(mat_G0, mean_cholesky_matrix.transpose()))
    
    return random_matrices

def scalars_gamma(n_samples, mean_value, dispersion_coefficient):
    gamma_shape = 1.0 / dispersion_coefficient**2
    gamma_scale = mean_value * dispersion_coefficient**2
    
    random_scalars = np.random.gamma(gamma_shape, gamma_scale, (n_samples,))
    
    return random_scalars