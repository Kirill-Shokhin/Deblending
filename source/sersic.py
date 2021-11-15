import numpy as np
from scipy.special import gamma, gammainc, gammaincinv

def b(n):
    return gammaincinv(2*n, 0.5) # сразу точный расчет через обратную ГФ

def flux_1d_sersic(n=1, r_e=50, max_=1):
    return 2*max_*n*r_e*gamma(n)/b(n)**(n)

def flux_2d_sersic(n=1, r_e=50, max_=1, ellip=0):
    return 2*np.pi*n*max_*r_e**2*gamma(2*n)*(1-ellip)/b(n)**(2*n)

def cdf_sersic(x, n=1, r_e=50):
    x = np.array(x)
    return 0.5*(1 + np.sign(x)*gammainc(n, b(n)*(np.abs(x)/r_e)**(1/n)))

def ppf_sersic(y, n=1, r_e=50):
    y= np.array(y)
    return np.sign(y-0.5)*r_e*(gammaincinv(n, 2*np.abs(y-0.5))/b(n))**n

def Sersic_1D(x, n=1, r_e=50, max_=1):
    x = np.array(x)
    return max_*np.exp(-b(n)*(np.abs(x)/r_e)**(1/n))

def Sersic_2D(center, theta=0, ellip=0, n=2, r_e=5, size = 100, max_ = 1):
    x0, y0 = center
    theta *= np.pi/180
    x = np.arange(size)
    y = x[:,np.newaxis]
    a, b = r_e, (1 - ellip) * r_e
    major = (x-x0) * np.cos(theta) + (y-y0) * np.sin(theta)
    minor = -(x-x0) * np.sin(theta) + (y-y0) * np.cos(theta)
    return max_*np.exp(-b_n*((np.hypot(major/a, minor/b))**(1/n)))