import direction_estimation as de
import numpy as np
import matplotlib.pyplot as plt


num_samples = 1024*64
num_antennas = 5
y = np.array([0, 0, 0, 0, 0])
antenna_distance = 0.175
x = np.array([0, 1, 2, 3, 4]) * antenna_distance
detection_range = 180
thetas = np.arange(-detection_range/2 - 90, detection_range/2 - 90)

iq_samples = (np.random.randn(num_antennas, num_samples) + 1j * np.random.randn(num_antennas, num_samples)).astype(np.complex64)

spatial_corr_matrix = de.spatial_correlation_matrix(iq_samples, num_samples)
spatial_corr_matrix = de.spatial_smoothing(num_antennas,iq_samples,3, "forward-backward")
#spatial_corr_matrix = de.forward_backward_avg(spatial_corr_matrix)
sig_dim = de.infer_signal_dimension(spatial_corr_matrix)
scanning_vectors = de.gen_scanning_vectors(num_antennas, x,y,thetas)
doa = de.DOA_MUSIC(spatial_corr_matrix, scanning_vectors, sig_dim)

# plt.figure()
# plt.plot(np.abs(doa))
# plt.title('DOA')
# plt.xlabel('Degree')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

