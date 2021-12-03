import pandas as pd
import numpy as np
import plotly.express as px
from Objects import Sersic, Moffat

class Create_data:
    def __init__(self, n_objects=100, size=200, lam=100, p=0.5, seed=None):
        self.n_objects = n_objects
        self.size = size
        self.lam = lam
        self.p = p
        self.seed = seed
        
    def cells_of_objects(self):
        cells = np.unique(self.rng.randint(0, self.size, (2, self.n_objects)), axis=1)
        for _ in range(5):       
            if cells.shape[1] < self.n_objects:
                pack = self.rng.randint(0, self.size, (2, self.n_objects-cells.shape[1]))
                cells = np.unique(np.concatenate((cells, pack), axis=1), axis=1)
            else:
                break
        return cells
        
    def generate_params(self):
        self.rng = np.random.RandomState(self.seed)
        cells = self.cells_of_objects()
        centers = self.rng.rand(2, self.n_objects)
        conf = np.ones(self.n_objects)
        class_ = self.rng.choice(2, self.n_objects, p=[1-self.p, self.p])# p - вероятность выпадения галактики 
        fwhm = np.abs(self.rng.normal(scale=3, size=(self.n_objects)))+1
        max_ = 10*np.sqrt(self.lam)*(1+self.rng.rand(self.n_objects))*fwhm
        
        beta = np.abs(self.rng.normal(scale=3, size=(self.n_objects-class_.sum())))+1.5
        n = 0.5 + self.rng.rand(class_.sum())*np.log(max_[class_==1])*fwhm[class_==1]/20
        beta_n = np.zeros(self.n_objects)
        beta_n[class_==0] = beta
        beta_n[class_==1] = n
        params = pd.DataFrame(np.array([class_, conf, *cells, *centers, fwhm, max_, beta_n]).T.astype('float32'), 
                columns=['class_', 'conf', 'cx', 'cy', 'x_in_cell', 'y_in_cell', 'fwhm', 'max_', 'beta / n'])
        params[params.columns[:4]] = params[params.columns[:4]].astype('uint16')
        params[['ltrb_box', 'w_box', 'h_box', 'flux']] = None
        return params

    def create_image(self, noise=True):
        self.rng = np.random.RandomState(self.seed)
        self.params = self.generate_params()
        img = np.zeros((self.size, self.size))
        for index, class_, conf, cx, cy, x_in_cell, y_in_cell, fwhm, max_, beta, _, _, _, _ in self.params.itertuples():
            if class_ == 0:
                obj = Moffat(beta, max_, x_in_cell, y_in_cell, fwhm=fwhm)
                x, y = (obj.r_int, obj.r_int)
                w_box, h_box = (2*obj.box_r, 2*obj.box_r)
            elif class_ == 1:
                obj = Sersic(beta, max_, 360*self.rng.rand(), self.rng.rand()/1.5, x_in_cell, y_in_cell, fwhm=fwhm, ratio=30)
                x, y = (int(obj.w/2), int(obj.h/2))    
                w_box, h_box = obj.w_box, obj.h_box
            self.params.loc[index, ('w_box', 'h_box')] = w_box, h_box
            self.params.at[index, 'ltrb_box'] = obj.cxcy2ltrb(cx+x_in_cell-0.5, cy+y_in_cell-0.5)
            self.params.loc[index, 'flux'] = obj.flux_in_box
            img[max(cy-y,0):min(cy+y+1, self.size),
                max(cx-x,0):min(cx+x+1, self.size)] += obj.make_2d()[max(y-cy,0):min(self.size-cy+y, 2*y+1),
                                                      max(x-cx,0):min(self.size-cx+x, 2*x+1)]
        if noise:
            img += self.poisson_noise()
        return img.astype('int32')
    
    def poisson_noise(self):
        self.rng = np.random.RandomState(self.seed)
        return self.rng.normal(self.lam, np.sqrt(self.lam), (self.size, self.size))
    
    def plot(self, bbox=True, noise=True):
        fig = px.imshow(self.create_image(noise))
        if bbox:
            for x1, y1, x2, y2 in self.params.ltrb_box:
                fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color='red'))
        return fig
    
    def data(self, columns=None, noise=True):
        img = self.create_image(noise)
        params = self.params.drop(['ltrb_box', 'cx', 'cy'], axis=1)
        if columns is None:
            params = params
        else:
            params = params[columns]
        return img, self.mask(), params.astype('float32').values
    
    def mask(self): # маска на изображение (cells)
        mask = np.zeros((self.size, self.size), dtype=bool)
        mask[self.params['cy'], self.params['cx']] = True
        return mask
    
class Create_dataset(Create_data):
    def __init__(self, n_images=1, n_objects=100, size=200, lam=[100, 10000], p=0.5, seed=None):
        self.n_images = n_images
        self.n_objects = n_objects
        self.size = size
        self.lam_bound = lam
        self.p = p
        self.seed = seed
    
    def make(self, columns=None):
        self.rng = np.random.RandomState(self.seed)
        dataset = []
        for lam in self.rng.uniform(self.lam_bound[0], self.lam_bound[1], self.n_images):
            self.lam = lam
            dataset.append(self.data(columns))
        return dataset