import numpy as np
from scipy.special import gamma, gammainc, gammaincinv, hyp2f1
from scipy.optimize import minimize
import plotly.express as px

class Moffat: #!!! beta > 1 !!!
    def __init__(self, beta=2, max_=1, x_in_cell=0.5, y_in_cell=0.5, ratio=3, box_ratio=1.5, r_e=None, fwhm=None):
        self.beta = beta # ~3-5
        self.max_ = max_
        self.x_in_cell = x_in_cell
        self.y_in_cell = y_in_cell
        self.ratio = ratio
        self.box_ratio = box_ratio
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
        self.r = self.box(ratio)
        self.r_int = int(np.ceil(self.r))
        self.box_r = self.box(box_ratio)
    
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
    
    def cdf_2d(self, r): # r > 0 
        return self.flux_2d*(1-(1+(r/self.r_e)**2)**(1-self.beta))
    
    @property 
    def flux_1d(self):
        return self.max_*self.r_e*np.sqrt(np.pi)*gamma(self.beta-0.5)/gamma(self.beta)
    
    @property     
    def flux_2d(self):
        return self.max_*np.pi*self.r_e**2/(self.beta-1)
    
    @property
    def flux_in_box(self):
        return self.cdf_2d(self.box_r) 
    
    def box(self, ratio=3):
        r = self.fwhm/2*ratio
        return r
    
    def make_1d(self):
        x = np.linspace(-self.r, self.r, 1000) 
        y = self.moffat(x)
        return x, y
    
    def make_2d(self):
        x = np.arange(2*self.r_int+1)
        y = x[:,np.newaxis]
        return self.moffat(np.hypot(x-self.r_int-(self.x_in_cell-0.5), y-self.r_int-(self.y_in_cell-0.5)))
    
    def plot_1d(self):
        fig = px.line(None, *self.make_1d())
        fig.add_vrect(x0=-self.box_r, x1=self.box_r, annotation_text="{}% fwhm".format(int(self.box_ratio*100)), annotation_position="top left", fillcolor="green", opacity=0.25, line_width=0)
        return fig
    
    def plot_2d(self, zmax=1):
        x1, y1, x2, y2 = self.cxcy2ltrb(self.r_int+self.x_in_cell-0.5, self.r_int+self.y_in_cell-0.5)
        fig = px.imshow(self.make_2d(), zmax=zmax*self.max_)
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color='red'))
        return fig
    
    def cxcy2ltrb(self, cx, cy):
        return cx-self.box_r, cy-self.box_r, cx+self.box_r, cy+self.box_r
    
   
    
class Sersic:
    def __init__(self, n=1, max_=1, theta=0, ellip=0, x_in_cell=0.5, y_in_cell=0.5,
                 ratio=4, box_ratio=2, r_e=None, fwhm=None):
        self.n = n
        self.max_ = max_
        self.theta = theta*np.pi/180
        self.ellip = ellip
        self.b_n = gammaincinv(2*n, 0.5) # сразу точный расчет через обратную ГФ
        self.x_in_cell = x_in_cell
        self.y_in_cell = y_in_cell
        self.ratio = ratio
        self.box_ratio = box_ratio
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
        self.w, self.h = self.box(ratio, discrete=True)
        self.w_box, self.h_box = self.box(box_ratio, discrete=False)
        
    def sersic(self, r):
        return self.max_*np.exp(-self.b_n*(np.abs(r)/self.r_e)**(1/self.n))
    
    def pdf(self, x):
        return self.sersic(x) / self.flux_1d
    
    def cdf(self, x): # 1D
        x = np.array(x)
        return 0.5*(1 + np.sign(x)*gammainc(self.n, self.b_n*(np.abs(x)/self.r_e)**(1/self.n)))

    def ppf(self, y): # 1D
        y = np.array(y)
        return np.sign(y-0.5)*self.r_e*(gammaincinv(self.n, 2*np.abs(y-0.5))/self.b_n)**self.n
    
    def cdf_2d(self, r): # r > 0
        return self.flux_2d*gammainc(2*self.n, self.b_n*(np.abs(r)/self.r_e)**(1/self.n))
    
    @property
    def flux_1d(self):
        return 2*self.max_*self.n*self.r_e*gamma(self.n)/self.b_n**(self.n)
    
    @property
    def flux_2d(self):
        return 2*np.pi*self.n*self.max_*self.r_e**2*gamma(2*self.n)*(1-self.ellip)/self.b_n**(2*self.n)
    
    @property
    def flux_in_box(self):
        return self.cdf_2d(self.box_r) 

    def box(self, ratio=1, discrete=False):
        r = self.fwhm/2*ratio
        w = 2*r*(1 - self.ellip*(1-np.abs(np.cos(self.theta))))
        h = 2*r*(1 - self.ellip*(1-np.abs(np.sin(self.theta))))
        if discrete:
            self.r = r
            w, h = (int(np.ceil(w)), int(np.ceil(h)))
            w, h = self.make_odd([w, h])
        else:
            self.box_r = r 
        return w, h
    
    def make_1d(self):
        x = np.linspace(-self.r, self.r, 1000) 
        y = self.sersic(x)
        return x, y
    
    def make_2d(self): # w, h нечетные, чтобы объект был в центральном пикселе
        x, y = self.rotation(np.arange(self.w)-self.w/2-self.x_in_cell+1,
                             (np.arange(self.h)-self.h/2-self.y_in_cell+1)[:, None], self.theta)
        r = np.hypot(x, y/(1-self.ellip))
        return self.sersic(r)
    
    def plot_1d(self):
        fig = px.line(None, *self.make_1d())
        fig.add_vrect(x0=-self.box_r, x1=self.box_r, annotation_text="{}% fwhm".format(int(self.box_ratio*100)),
                      annotation_position="top left", fillcolor="green", opacity=0.25, line_width=0)
        return fig
    
    def plot_2d(self, zmax=0.5):
        img = self.make_2d()
        x1, y1, x2, y2 = self.cxcy2ltrb(self.w/2+self.x_in_cell-1, self.h/2+self.y_in_cell-1)
        fig = px.imshow(img, zmax=zmax*self.max_)
        fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color='red'))
        return fig
    
    def rotation(self, x, y, theta):
        x1 = x*np.cos(theta) + y*np.sin(theta)
        y1 = -x*np.sin(theta) + y*np.cos(theta)
        return x1, y1
    
    def cxcy2ltrb(self, cx, cy):
        return cx-self.w_box/2, cy-self.h_box/2, cx+self.w_box/2, cy+self.h_box/2
    
    def make_odd(self, x):
        x = np.array(x)
        x[x%2==0] += 1
        return x