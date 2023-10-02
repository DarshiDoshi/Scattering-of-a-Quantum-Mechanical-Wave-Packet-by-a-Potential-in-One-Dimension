import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Parameters
N = 1024 # number of grid points
L = 100 # grid length
x = np.linspace(-L/2, L/2, N) # position grid
dx = x[1] - x[0] # grid spacing
dt = 0.1 # time step
tmax = 10 # maximum time
timesteps = int(tmax/dt) # number of time steps

# Initial wavefunction
x0 = 0 # center of wavepacket
sigma = 1 # width of wavepacket
k0 = 5 # initial momentum
psi = np.exp(-(x-x0)**2/(2*sigma**2))*np.exp(1j*k0*x) # initial wavefunction

# Potential energy
V = np.zeros(N) # no potential

# Kinetic energy operator in momentum space
dk = 2*np.pi/L
k = np.concatenate((np.arange(0,N/2+1)*dk,np.arange(-N/2+1,0)*dk))
T = np.exp(-1j*(k**2)*dt/2)

# Potential energy operator in position space
U = np.exp(-1j*V*dt)

# Time evolution
for t in range(timesteps):
    psi = ifft(T*fft(U*psi)) # split operator method
    plt.plot(x,np.abs(psi)**2)
    plt.xlabel('Position (x)')
    plt.ylabel('Probability Density (psi*psi)')
    plt.axis([-L/2,L/2,0,1])
    plt.draw()
    plt.pause(0.01)
    plt.clf()
