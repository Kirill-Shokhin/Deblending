import numpy as np
from numpy.random import normal, random, poisson, uniform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# Генерируем пиксели, в которых будут объекты.
# Цикл нужен, потому как при длине вектора более 300, иногда появляются повторяющиеся ячейки
# а нам нужно соотвествие n_objects и len(cells)
def cells_of_objects(n_objects, size=100): #до 300 объектов
    while True:
        cells = np.unique(np.random.randint(0, size, (2, n_objects)), axis=1)
        if cells.shape[1]==n_objects:
            break
    return cells

# Радиус гауссианы, содержащий 0.95 потока
# Также используется при создании самой гауссианы, для выбора размера массива (0.99 потока)
def radius_gauss(fwhm, ratio=0.95):
    if ratio == 0.95:
        ppf = 1.959963984540054
    elif ratio == 0.99:
        ppf = 2.5758293035489004
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    r = ppf*sigma
    return r


# Генерируем все остальные параметры гауссианы
# 2 - center of objects, 1 - confidence, 2 - params (flux, fwhm), 2 - size of bbox (h, w)
# ! поток зависит от параметра шума лямбды !
def generate_params(n_objects, lam):
    params = np.zeros((6, n_objects))
    centers = random((2, n_objects))
    conf = np.ones(n_objects)
    fwhm = np.abs(normal(scale=3, size=(n_objects)))+1
    flux = 10*np.sqrt(lam)*(1+random(n_objects))*fwhm**2
    r = radius_gauss(fwhm, ratio=0.95)
    params[:] = *centers, conf, fwhm, flux, r
    return params.astype('float32').T

# Создаем одно изображение по готовым параметрам
def create_objects(params, cells, size):
    img = np.zeros((size,size))
    for cx, cy, x_in_cell, y_in_cell, _, fwhm, flux, _ in np.concatenate((cells.T, params), axis=1): #написать класс: через add
        cx, cy = (int(cx), int(cy))
        sigma = fwhm/(2*np.sqrt(2*np.log(2))) 
        obj, r = makeGauss(x_in_cell, y_in_cell, fwhm, flux)
        img[max(cx-r,0):min(cx+r+1, size),
            max(cy-r,0):min(cy+r+1, size)] += obj[max(r-cx,0):min(size-cx+r, 2*r+1),
                                                  max(r-cy,0):min(size-cy+r, 2*r+1)]
    return img.T

# Генерируем гауссиану на маленьком изображении радиуса, содержащего 0.99 потока + округление
def makeGauss(x_in_cell, y_in_cell, fwhm, flux):
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    r = np.floor(radius_gauss(fwhm, ratio=0.99))+1
    x = np.arange(2*r+1)
    y = x[:,np.newaxis]
    obj = flux*np.exp(-1/2*((x-r-(x_in_cell-0.5))**2 + (y-r-(y_in_cell-0.5))**2) / sigma**2)/(2*np.pi*sigma**2)
    return obj, int(r)


def poisson_noise(lam, size=200):
    return normal(lam, np.sqrt(lam), (size,size))

# Создаем сет: изображение и тензор параметров
def create_data(size=200, n_objects=150, lam=500, noise=True):
    cells = cells_of_objects(n_objects, size)
    params = generate_params(n_objects, lam)
    
    gt = np.zeros((size, size, params.shape[1])) #gt - ground truth
    gt[cells[0], cells[1]] = params
    
    img = create_objects(params, cells, size)
    if noise:
        img += poisson_noise(lam, size)
    return gt.astype('float32').T, img.astype(int)[None, :]

# Создаем датасет. 
# Параметры: size - кол-во сетов и px_size - размер изображения
def create_Dataset(size=512, px_size=200):
    dataset = []
    for lam in uniform(100, 10000, size):
        dataset.append(create_data(size=px_size, lam=lam))
    return dataset


# Далее функции занимаются отрисовкой рамок на изображениях.
def convert_to_box(x, y, r):
    return np.concatenate([[x-r], [y-r], [x+r], [y+r]], axis=0).T

def bboxs(params, mask):
    cx, cy = np.where(mask>0)
    x_in_cell, y_in_cell, r = params[:, cx, cy][[0,1,5]]
    x, y  = (x_in_cell+cx-0.5, y_in_cell+cy-0.5)
    return convert_to_box(y, x, r)

def Mask(conf, treshhold=0.5):
    return conf>treshhold

def plot_bboxs(img, num=0, output=None, gt=None, mask=None):
    if 'torch' in str(img.dtype):           
        img = img[num].cpu().detach().numpy()
        if output is not None:
            output = output[num].cpu().detach().numpy() 
        if gt is not None:
            gt = gt[num].cpu().detach().numpy()
        if mask is not None:
            mask = mask[num].cpu().detach().numpy()
    
    fig = px.imshow(np.array((img[0], img[0])), facet_col=0)
    if output is not None:    
        if mask is None:
            mask = gt[2]>0
        for x0, y0, x1, y1 in bboxs(output, mask):
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color='red'), row=1, col=1)
    if gt is not None:
        for x0, y0, x1, y1 in bboxs(gt, gt[2]>0):
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color='red'), row=1, col=2)
    fig.layout.annotations[0].text = 'Prediction'
    fig.layout.annotations[1].text = 'Ground Truth'
    return fig