import numpy as np
from numpy import linalg as lin
from numba import njit
from functools import lru_cache

#This file contains numba optimized code from the pyargus library

@njit(fastmath=True, cache=True)
def spatial_correlation_matrix(samples, num_samples):
    """
    Computes the spatial correlation matrix for a given set of samples.

    Parameters:
    -----------
    samples : ndarray
        A 2D array where each column represents a sample.
    num_samples : int
        The number of samples.

    Returns:
    --------
    spatial_corr_matrix : ndarray
        The computed spatial correlation matrix.
    """
    samples = np.ascontiguousarray(samples)
    spatial_corr_matrix = np.dot(samples, samples.conj().T)
    spatial_corr_matrix = np.divide(spatial_corr_matrix, num_samples).astype(np.complex64)
    return spatial_corr_matrix

@njit(fastmath=True, cache=True)
def DOA_MUSIC(R, scanning_vectors, signal_dimension):
    """
    Estimates the Direction of Arrival (DOA) using the MUSIC algorithm.

    Parameters:
    -----------
    R : ndarray
        Spatial correlation matrix.
    scanning_vectors : ndarray
        Array of scanning vectors.
    signal_dimension : int
        Number of signal sources.

    Returns:
    --------
    ADORT : ndarray
        Array of DOA estimates. If input dimensions are incorrect, returns an array with a single element (-1 or -2).
    """
    # --> Input check
    if R[:, 0].size != R[0, :].size:
        print("ERROR: Correlation matrix is not quadratic")
        return np.ones(1, dtype=np.complex64) * -1  

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

@njit(fastmath=True, cache=True)
def forward_backward_avg(R):
    """
    Computes the forward-backward averaged spatial correlation matrix.

    Parameters:
    -----------
    R : ndarray
        Spatial correlation matrix.

    Returns:
    --------
    R_fb : ndarray
        The forward-backward averaged spatial correlation matrix.
    """
    # --> Calculation
    M = R[:, 0].size 
    
    # Create exchange matrix
    J = np.eye(M, dtype=np.complex64)
    J = np.ascontiguousarray(np.fliplr(J))

    R_fb = 0.5 * (R + J@np.conjugate(R)@J)

    return np.ascontiguousarray(R_fb)

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
        theta_rad = np.deg2rad(thetas[i])        
        scanning_vectors[:,i] = np.exp(1j*2*np.pi* (x*np.cos(theta_rad) + y*np.sin(np.deg2rad(theta_rad))))    
    
    return np.ascontiguousarray(scanning_vectors)


@lru_cache(maxsize=32)
@njit(fastmath=True, cache=True)
def gen_scanning_vectors_optimized(M, array_type, antenna_distance, detection_range):
    """
    Generates scanning vectors for a uniform linear array (ULA) configuration.
    This function is suitable for use in situations where scanning vectors are dynamic.

    Parameters:
    -----------
    M : int
        Number of elements in the antenna array.
    array_type : str
        Type of antenna array. Currently, only "ULA" (Uniform Linear Array) is supported.
    antenna_distance : float
        Distance between adjacent antennas in the array.
    detection_range : float
        Range of angles to scan for detection, in degrees.

    Returns:
    --------
    scanning_vectors : ndarray
        A 2D array where each column represents a scanning vector corresponding to a specific angle.

    """

    if array_type == "ULA":
        y = np.array([0, 0, 0, 0, 0])
        x = np.array([0, 1, 2, 3, 4]) * antenna_distance
    else:
        raise RuntimeError("Unsupported array type, use unoptimized version")
    
    thetas = np.arange(-detection_range/2 - 90, detection_range/2 - 90)
    
    scanning_vectors = np.zeros((M, thetas.size), dtype=np.complex64)
    for i in range(thetas.size):
        theta_rad = np.deg2rad(thetas[i])        
        scanning_vectors[:,i] = np.exp(1j*2*np.pi* (x*np.cos(theta_rad) + y*np.sin(np.deg2rad(theta_rad))))    
    
    return np.ascontiguousarray(scanning_vectors)


@njit(fastmath=True, cache=True)
def spatial_smoothing(M, iq_samples ,P, direction): 
    """
    Performs spatial smoothing on input IQ samples to improve signal processing in scenarios with coherent sources.

    Parameters:
    -----------
    M : int
        Number of elements in the antenna array.
    iq_samples : ndarray
        2D array of IQ samples where each column represents a sample at a specific time.
    P : int
        Number of elements in each subarray for smoothing.
    direction : str
        Direction of smoothing, can be "forward", "backward", or "forward-backward".

    Returns:
    --------
    Rss : ndarray
        The spatially smoothed correlation matrix.

    """

    N = iq_samples[0, :].size
    L = M - P + 1  #Number of subarrays    

    Rss = np.zeros((P, P), dtype=np.complex64)  
    if direction == "forward" or direction == "forward-backward":            
        for l in range(L):             
            Rxx = np.zeros((P,P), dtype=np.complex64) # Correlation matrix allocation 
            for n in np.arange(0,N,1): 
                Rxx += np.outer(iq_samples[l:l+P,n], np.conj(iq_samples[l:l+P,n])) 
            np.divide(Rxx,N) # normalization 
            Rss += Rxx
    if direction == "backward" or direction == "forward-backward":         
        for l in range(L): 
            Rxx = np.zeros((P, P), dtype=np.complex64)  # Correlation matrix allocation 
            for n in np.arange(0,N,1): 
                d = np.conj(iq_samples[M-l-P:M-l, n][::-1]) 
                Rxx += np.outer(d, np.conj(d)) 
            np.divide(Rxx,N)  # normalization 
            Rss += Rxx 
    # normalization            
    if direction == "forward-backward": 
        np.divide(Rss, 2*L)  
    else: 
        np.divide(Rss,L)  

    return np.ascontiguousarray(Rss)

@njit(fastmath=True, cache=True)
def infer_signal_dimension(correlation_matrix, threshold_ratio=0.3):
    """
    Infers the signal dimension (number of signals) based on the eigenvalues of the correlation matrix.

    Parameters:
    -----------
    correlation_matrix : ndarray
        The input correlation matrix.
    threshold_ratio : float, optional
        Ratio of the threshold for determining signal dimension.

    Returns:
    --------
    signal_dimension : int
        The inferred number of signals, capped at a maximum of 4.
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    
    # Compute magnitudes of eigenvalues
    magnitudes = np.abs(eigenvalues)
    
    # Sort magnitudes in descending order
    sorted_magnitudes = -np.sort(-magnitudes) 
    
    # Determine the threshold
    threshold = threshold_ratio * sorted_magnitudes[0]
    
    # Find the number of eigenvalues greater than the threshold
    signal_dimension = np.sum(sorted_magnitudes > threshold)
    
    return min(signal_dimension,4)