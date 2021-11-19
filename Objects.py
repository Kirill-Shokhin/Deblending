import numpy as np
from scipy.special import gamma, gammainc, gammaincinv, hyp2f1
from scipy.optimize import minimize
import plotly.express as px

class Moffat: #!!! beta > 1 !!!
    def __init__(self, beta=2, max_=1, r_e=None, fwhm=None):
        self.beta = beta # ~3-5
        self.max_ = max_
        if r_e is not None and fwhm is not None:
            raise ValueError('Сhoose only one: fwhm or r_e')
        if r_e is not None:
            self.r_e = r_e 
            self.fwhm = 2*r_e*np.sqrt(2**(1/beta)-1) 
        elif fwhm is not None:
            self.fwhm = fwhm
            self.r_e = fwhm/(2*np.sqrt(2**(1/beta)-1))
        else:
            raise ValueError('Missing the 1 required parameter: fwhm or r_e')
    
    def moffat(self, r):
        return self.max_*(1 + (r/self.r_e)**2)**(-self.beta)
    
    def pdf(self, x):
        x = np.array(x)
        return self.moffat(x) / self.flux_1d
    
    def cdf(self, x):
        x = np.array(x)
        return self.max_*x*hyp2f1(0.5, self.beta, 1.5, -(x/self.r_e)**2)/self.flux_1d + 0.5
    
    def ppf(self, y):
        y = np.array(y)
        fun = lambda x: ((self.cdf(x)-y)**2).mean()
        return minimize(fun, x0=np.zeros_like(y), tol=1e-10).x
    
    @property 
    def flux_1d(self):
        return self.max_*self.r_e*np.sqrt(np.pi)*gamma(self.beta-0.5)/gamma(self.beta)
    
    @property     
    def flux_2d(self):
        return self.max_*np.pi*self.r_e**2/(self.beta-1)
    
    def box(self, ratio=3):
        r = self.fwhm/2*ratio
        return r
    
    def make_1d(self, ratio=3):
        self.r = self.box(ratio)
        x = np.linspace(-self.r, self.r, 1000) 
        y = self.moffat(x)
        return x, y
    
    def make_2d(self, x_in_cell=0.5, y_in_cell=0.5, ratio=3):
        self.r = np.floor(self.box(ratio))+1
        x = np.arange(2*self.r+1)
        y = x[:,np.newaxis]
        return self.moffat(np.hypot(x-self.r-(x_in_cell-0.5), y-self.r-(y_in_cell-0.5)))
    
    def plot_1d(self, ratio=3, box_ratio=1.5):
        self.box_r = self.box(box_ratio)
        fig = px.line(None, *self.make_1d(ratio))
        fig.add_vrect(x0=-self.box_r, x1=self.box_r, annotation_text="{}% fwhm".format(int(box_ratio*100)), annotation_position="top left", fillcolor="green", opacity=0.25, line_width=0)
        return fig
    
    def plot_2d(self, x_in_cell=0.5, y_in_cell=0.5, ratio=3, box_ratio=1.5, zmax=1):
        img = self.make_2d(x_in_cell, y_in_cell, ratio)
        self.box_r = self.box(box_ratio)
        x1, y1, x2, y2 = self.cxcy2ltrb(self.r+x_in_cell-0.5, self.r+y_in_cell-0.5, self.box_r)
        fig = px.imshow(img, zmax=zmax*self.max_)
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color='red'))
        return fig
    
    def cxcy2ltrb(self, x, y, box_r):
        return x-box_r, y-box_r, x+box_r, y+box_r
    
   
    
class Sersic:
    def __init__(self, n=1, max_=1, theta=0, ellip=0, r_e=None, fwhm=None):
        self.n = n
        self.max_ = max_
        self.theta = theta*np.pi/180
        self.ellip = ellip
        self.b_n = gammaincinv(2*n, 0.5) # сразу точный расчет через обратную ГФ
        if r_e is not None and fwhm is not None:
            raise ValueError('Сhoose only one: fwhm or r_e')
        if r_e is not None:
            self.r_e = r_e
            self.fwhm = 2*r_e*(np.log(2)/self.b_n)**n 
        elif fwhm is not None:
            self.fwhm = fwhm
            self.r_e = fwhm*(self.b_n/np.log(2))**n / 2
        else:
            raise ValueError('Missing the 1 required parameter: fwhm or r_e')
        
    def sersic(self, r):
        return self.max_*np.exp(-self.b_n*(np.abs(r)/self.r_e)**(1/self.n))
    
    def pdf(self, x):
        return self.sersic(x) / self.flux_1d
    
    def cdf(self, x): # 1D
        x = np.array(x)
        return 0.5*(1 + np.sign(x)*gammainc(self.n, self.b_n*(np.abs(x)/self.r_e)**(1/self.n)))

    def ppf(self, y): # 1D
        y= np.array(y)
        return np.sign(y-0.5)*self.r_e*(gammaincinv(self.n, 2*np.abs(y-0.5))/self.b_n)**self.n
    
    @property
    def flux_1d(self):
        return 2*self.max_*self.n*self.r_e*gamma(self.n)/self.b_n**(self.n)
    
    @property
    def flux_2d(self):
        return 2*np.pi*self.n*self.max_*self.r_e**2*gamma(2*self.n)*(1-self.ellip)/self.b_n**(2*self.n)
    
    def box(self, ratio=0.95, discrete=False):
        r = self.fwhm/2*ratio
        w = 2*r*(1 - self.ellip*(1-np.abs(np.cos(self.theta))))
        h = 2*r*(1 - self.ellip*(1-np.abs(np.sin(self.theta))))
        if discrete:
            w, h = (np.floor(w)+1, np.floor(h)+1)
        return w, h
    
    def make_1d(self, ratio=2):
        self.r = self.fwhm/2*ratio
        x = np.linspace(-self.r, self.r, 1000) 
        y = self.sersic(x)
        return x, y
    
    def make_2d(self, x_in_cell=0.5, y_in_cell=0.5, ratio=2):
        self.W, self.H = self.box(ratio, discrete=True)
        x, y = self.rotation(np.arange(2*self.W+1)-self.W-x_in_cell+0.5,
                             (np.arange(2*self.H+1)-self.H-y_in_cell+0.5)[:, None], self.theta)
        r = np.hypot(x, y/(1-self.ellip))
        return self.sersic(r)
    
    def plot_1d(self, ratio=2, box_ratio=1):
        self.box_r = self.fwhm/2*box_ratio
        fig = px.line(None, *self.make_1d(ratio))
        fig.add_vrect(x0=-self.box_r, x1=self.box_r, annotation_text="{}% fwhm".format(int(box_ratio*100)),
                      annotation_position="top left", fillcolor="green", opacity=0.25, line_width=0)
        return fig
    
    def plot_2d(self, x_in_cell=0.5, y_in_cell=0.5, ratio=2, box_ratio=1, zmax=0.5):
        img = self.make_2d(x_in_cell, y_in_cell, ratio)
        self.w, self.h = self.box(box_ratio)
        x1, y1, x2, y2 = self.cxcy2ltrb(self.W+x_in_cell-0.5, self.H+y_in_cell-0.5, self.w, self.h)
        fig = px.imshow(img, zmax=zmax*self.max_)
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color='red'))
        return fig
    
    def rotation(self, x, y, theta):
        x1 = x*np.cos(theta) + y*np.sin(theta)
        y1 = -x*np.sin(theta) + y*np.cos(theta)
        return x1, y1
    
    def cxcy2ltrb(self, cx, cy, w, h):
        return cx-w, cy+h, cx+w, cy-h