from numpy.random import normal, random
import pandas as pd
import numpy as np
import plotly.express as px
from Objects import Sersic, Moffat

class Create_data:
    def __init__(self, n_objects=100, size=200, lam=100, p=0.5):
        self.n_objects = n_objects
        self.size = size
        self.lam = lam
        self.p = p
        self.params = self.generate_params()
        
    def cells_of_objects(self):
        cells = np.unique(np.random.randint(0, self.size, (2, self.n_objects)), axis=1)
        for _ in range(5):       
            if cells.shape[1] < self.n_objects:
                pack = np.random.randint(0, self.size, (2, self.n_objects-cells.shape[1]))
                cells = np.unique(np.concatenate((cells, pack), axis=1), axis=1)
            else:
                break
        return cells
        
    def generate_params(self):
        cells = self.cells_of_objects()
        centers = random((2, self.n_objects))
        conf = np.ones(self.n_objects)
        class_ = np.random.choice(2, self.n_objects, [1-self.p, self.p]) # p - вероятность выпадения галактики 
        fwhm = np.abs(normal(scale=3, size=(self.n_objects)))+1
        max_ = 10*np.sqrt(self.lam)*(1+random(self.n_objects))*fwhm
        
        beta = np.abs(normal(scale=3, size=(self.n_objects-class_.sum())))+1.5
        n = 0.5 + random(class_.sum())*np.log(max_[class_==1])*fwhm[class_==1]/20
        beta_n = np.zeros(self.n_objects)
        beta_n[class_==0] = beta
        beta_n[class_==1] = n
        
        params = pd.DataFrame(np.array([class_, conf, *cells, *centers, fwhm, max_, beta_n]).T.astype('float32'), 
                columns=['class_', 'conf', 'cx', 'cy', 'x_in_cell', 'y_in_cell', 'fwhm', 'max_', 'beta / n'])
        params[params.columns[:4]] = params[params.columns[:4]].astype('uint16')
        return params

    def create_image(self, noise=True):
        self.params['ltrb_box'] = None
        img = np.zeros((self.size, self.size))
        for index, class_, conf, cx, cy, x_in_cell, y_in_cell, fwhm, max_, beta, _ in self.params.itertuples():
            if class_ == 0:
                obj = Moffat(beta, max_, x_in_cell, y_in_cell, fwhm=fwhm)
                x, y = (obj.r_int, obj.r_int)
                w_box, h_box = (2*obj.box_r, 2*obj.box_r)
            elif class_ == 1:
                obj = Sersic(beta, max_, 360*random(), random()/1.5, x_in_cell, y_in_cell, fwhm=fwhm, ratio=30)
                y, x = (int(obj.w/2), int(obj.h/2))    
                w_box, h_box = obj.w_box, obj.h_box
            self.params.loc[index, ('w_box', 'h_box')] = w_box, h_box
            self.params.at[index, 'ltrb_box'] = obj.cxcy2ltrb(cy+x_in_cell-0.5, cx+y_in_cell-0.5)
            img[max(cx-x,0):min(cx+x+1, self.size),
                max(cy-y,0):min(cy+y+1, self.size)] += obj.make_2d()[max(x-cx,0):min(self.size-cx+x, 2*x+1),
                                                      max(y-cy,0):min(self.size-cy+y, 2*y+1)]
        if noise:
            img += self.poisson_noise()
        return img
    
    def poisson_noise(self):
        return normal(self.lam, np.sqrt(self.lam), (self.size, self.size))
    
    def plot(self, bbox=True, noise=True):
        fig = px.imshow(self.create_image(noise))
        if bbox:
            for x1, y1, x2, y2 in self.params.ltrb_box:
                fig.add_shape(type="rect", x0=x2, y0=y2, x1=x1, y1=y1, line=dict(color='red'))
        return fig