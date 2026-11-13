"""Window-local OHLC geometry; one shared price axis, optional volume panel."""
import numpy as np


def render_window(window,include_volume=True):
    x=np.asarray(window,dtype=float)
    if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all() or (x<=0).any():
        raise ValueError('Expected a complete positive OHLCV window')
    if not ((x[:,1]>=x[:,:4].max(axis=1)).all() and (x[:,2]<=x[:,:4].min(axis=1)).all()):
        raise ValueError('Invalid OHLC geometry')
    length=len(x);size=3*length
    volume_height=max(1,size//5) if include_volume else 0
    price_height=size-volume_height-(1 if include_volume else 0)
    image=np.zeros((size,size),dtype=np.uint8)
    low,high=float(x[:,2].min()),float(x[:,1].max())
    if high==low:
        y=np.full((length,4),price_height//2,dtype=int)
    else:
        y=price_height-1-np.rint((x[:,:4]-low)/(high-low)*(price_height-1)).astype(int)
    for i,(o,h,l,c) in enumerate(y):
        image[o,i*3]=255
        image[h:l+1,i*3+1]=255
        image[c,i*3+2]=255
        if include_volume:
            height=int(np.rint(x[i,4]/x[:,4].max()*volume_height))
            if height:
                image[size-height:,i*3+1]=180
    return image


def prediction_basis(window,multipliers):
    """Cancel end-of-dataset scale using the input endpoint's own multipliers."""
    x=np.array(window,dtype=float,copy=True)
    price,volume=np.asarray(multipliers,dtype=float)
    if not np.isfinite([price,volume]).all() or min(price,volume)<=0:
        raise ValueError('Invalid prediction-time basis')
    x[:,:4]/=price;x[:,4]/=volume
    return x
