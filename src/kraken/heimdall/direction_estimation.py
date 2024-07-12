import numpy as np
from numpy import linalg as lin
from numba import njit
from functools import lru_cache

@njit(fastmath=True, cache=True)
def spatial_correlation_matrix(samples, num_samples):
    samples = np.ascontiguousarray(samples)
    spatial_corr_matrix = np.dot(samples, samples.conj().T)
    spatial_corr_matrix = np.divide(spatial_corr_matrix, num_samples)
    #spatial_corr_matrix = forward_backward_avg(spatial_corr_matrix)
    return spatial_corr_matrix

@njit(fastmath=True, cache=True)
def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution=1):
    # --> Input check
    if R[:, 0].size != R[0, :].size:
        print("ERROR: Correlation matrix is not quadratic")
        return np.ones(1, dtype=np.complex64) * -1  # [(-1, -1j)]

    if R[:, 0].size != scanning_vectors[:, 0].size:
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return np.ones(1, dtype=np.complex64) * -2

    ADORT = np.zeros(scanning_vectors[0, :].size, dtype=np.complex64)
    M = R[:, 0].size  

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    sigmai = np.abs(sigmai)

    idx = sigmai.argsort()[::1]  # Sort eigenvectors by eigenvalues, smallest to largest
    vi = vi[:, idx]

    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension

    E = np.empty((M, noise_dimension), dtype=np.complex64)
    for i in range(noise_dimension):
        E[:, i] = vi[:, i]

    E_ct = E @ E.conj().T
    theta_index = 0
    for i in range(scanning_vectors[0, :].size):
        S_theta_ = scanning_vectors[:, i]
        S_theta_ = np.ascontiguousarray(S_theta_.T)
        ADORT[theta_index] = 1 / np.abs(S_theta_.conj().T @ E_ct @ S_theta_)
        theta_index += 1

    return ADORT

#TODO optimize
def forward_backward_avg(R):
    """
        Calculates the forward-backward averaging of the input correlation matrix
        
    Parameters:
    -----------
        :param R : Spatial correlation matrix
        :type  R : M x M complex numpy array, M is the number of antenna elements.        
            
    Return values:
    -------------
    
        :return R_fb : Forward-backward averaged correlation matrix
        :rtype R_fb: M x M complex numpy array           
        
        :return -1, -1: Input spatial correlation matrix is not quadratic
            
    """          
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1 
    
    # --> Calculation
    M = np.size(R, 0)  # Number of antenna elements
    R = np.matrix(R)

    # Create exchange matrix
    J = np.eye(M)
    J = np.fliplr(J) 
    J = np.matrix(J)
    
    R_fb = 0.5 * (R + J*np.conjugate(R)*J)

    return np.array(R_fb)

#TODO FIX LRU CACHE
#@lru_cache(maxsize=32)
@njit(fastmath=True, cache=True)
def gen_scanning_vectors(M, x, y, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for general antenna array configurations        
        
    Parameters:
    -----------

        :param M : Number of antenna elements on the circle
        :param x : x coordinates of the antenna elements on a plane
        :param y : y coordinates of the antenna elements on a plane
        :param thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]
        
        :type M: int
        :type x: 1D numpy array
        :type y: 1D numpy array
        :type R: float
        :type thetas: 1D numpy array
            
    Return values:
    -------------
    
        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles
        
    """
    scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
    for i in range(thetas.size):        
        scanning_vectors[:,i] = np.exp(1j*2*np.pi* (x*np.cos(np.deg2rad(thetas[i])) + y*np.sin(np.deg2rad(thetas[i]))))    
    
    return np.ascontiguousarray(scanning_vectors)


