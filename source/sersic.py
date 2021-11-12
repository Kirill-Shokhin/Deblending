from numpy import zeros, array

MIN_SIZE = 1e-6

def sersic(radius: int, 
           Ib:float = 60, 
           alpha: float = 1, 
           beta: float = 0.3, 
           gamma:float = 0.5, 
           rb:float = 11, 
           phi:float = 0) -> array:
    """
    Create a galaxy with the Sersic intensity profile (see, e.g. 
    https://iopscience.iop.org/article/10.1086/375320/pdf, page 1, formula 1)
    
    I(r) = I_b * 2^[(beta - gamma)/alpha] * (r/rb)^(-gamma) *
         * {1 + (r/rb)^alpha}^[(gamma-beta)/alpha]
         
    :params:
        radius: float - galactic radius in scale units (e.g. pixels or arcseconds)
    :keywords:
        Ib: float - peak itensity of the galactic core (default to 60 counts)
        alpha: float - transition parameter, controlling the smoothing between the two powerlaws (default to 0.3)
        beat: float 
    """
    
    # TODO:
    # Even kernel generation
    # Rotation and assymetry
    # Spirals
    
    galaxy_image = np.zeros((r, r))
    x0 = y0 = r // 2
    # Galactic core
    for x in range(-x0, x0+1):
        for y in range(-y0, y0+1):
            # Trace a circluar area, not the rectangular
            if np.hypot(x, y) > x0:
                continue
            bga = (beta-gamma)/alpha
            gba = -bga
            rrb = (x**2 + y**2) / rb
            rrb = MIN_SIZE if rrb < MIN_SIZE else rrb
            galaxy_image[x+x0,y+y0] = Ib*2**(bga)*(rrb)**(-gamma)
            galaxy_image[x+x0,y+y0] *= (1 + rrb**alpha)**gba
            
    return galaxy_image
