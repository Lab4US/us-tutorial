import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# --- 1. PARAMETERS ---
c = 1520              
fs = 100e6            
depth_max = 0.06      
t = np.arange(0, (2 * depth_max) / c, 1/fs)

num_channels = 128    
x_array = np.linspace(-0.015, 0.015, num_channels) 

# --- 2. PHANTOM GENERATION (Speckle + Point Targets + Large Targets) ---
num_speckle = 2000    
x_speckle = np.random.uniform(-0.02, 0.02, num_speckle)
z_speckle = np.random.uniform(0.01, depth_max, num_speckle)
amp_speckle = np.random.normal(0, 0.45, num_speckle) 

# A. Point targets (Punti singoli brillanti)
x_points = [0.0, -0.005, 0.005]
z_points = [0.03, 0.04, 0.04]   
amp_points = [6.0, 6.0, 6.0]  

# B. Large Target 1: Massa circolare (es. a sinistra)
num_pts_large = 100
theta = np.random.uniform(0, 2*np.pi, num_pts_large)
radius = np.random.uniform(0, 0.003, num_pts_large) # Raggio 3mm
x_large1 = -0.01 + radius * np.cos(theta)
z_large1 = 0.02 + radius * np.sin(theta)
amp_large1 = [3.0] * num_pts_large # Meno brillanti dei punti singoli per realismo

# C. Large Target 2: Struttura piatta/ellittica (es. a destra)
x_large2 = 0.01 + (radius * 2) * np.cos(theta) # Più larga
z_large2 = 0.02 + (radius * 0.5) * np.sin(theta) # Più sottile
amp_large2 = [3.0] * num_pts_large

# Combine everything
x_s = np.concatenate([x_speckle, x_points, x_large1, x_large2])
z_s = np.concatenate([z_speckle, z_points, z_large1, z_large2])
amps = np.concatenate([amp_speckle, amp_points, amp_large1, amp_large2])

# --- 3. IMPULSE e(t) ---
f0 = 5e6
pulse_t = np.arange(-1e-6, 1e-6, 1/fs)
et = np.exp(-0.5 * (pulse_t / 0.15e-6)**2) * np.cos(2 * np.pi * f0 * pulse_t)

# --- 4. SIMULATION ---
b_mode_matrix = np.zeros((len(t), num_channels))

print("Simulating scan lines...")
for i, x_tx in enumerate(x_array):
    rf_line = np.zeros_like(t)
    for xs, zs, amp in zip(x_s, z_s, amps):
        dist = np.sqrt((x_tx - xs)**2 + zs**2)
        tau = (2 * dist) / c
        idx_tau = np.argmin(np.abs(t - tau))
        start_idx = idx_tau - len(et)//2
        end_idx = start_idx + len(et)
        
        if 0 <= start_idx and end_idx < len(rf_line):
            lateral_weight = np.exp(-((x_tx - xs)**2) / (0.0005**2))
            rf_line[start_idx:end_idx] += et * amp * lateral_weight
            
    b_mode_matrix[:, i] = np.abs(hilbert(rf_line))

# --- 5. VISUALIZATION ---
plt.figure(figsize=(7, 9))
image_data = 20 * np.log10(b_mode_matrix + 1e-6) 
image_data -= np.max(image_data)

plt.imshow(image_data, aspect='auto', cmap='gray', vmin=-40, vmax=0,
           extent=[x_array[0]*100, x_array[-1]*100, depth_max*100, 0])

plt.title("Ultrasound B-mode: Multiple Target Sizes")
plt.xlabel("X Position (cm)")
plt.ylabel("Depth Z (cm)")
plt.colorbar(label="Intensity (dB)")
plt.show()