import torch
import torch.nn.functional as F

import numpy as np
import PIL.Image as pi
from osgeo import gdal,osr

def QD(U, L, x, lam, alpha, s, hard):
    
    k_h = (L <= x) * (x <= U)  
    
    if hard:
        k = (L <= x) * (x <= U)  
    else:
        k = torch.sigmoid((U-x  )*s) * torch.sigmoid((x-L)*s)
        
    PICP = k.mean()
    MPIW = torch.sum((U - L)*k_h) / (torch.sum(k_h) + 1e-7)
    
    loss = MPIW + lam * (F.relu(1-alpha-PICP))**2 
    
    # to stablize the training procedure
    loss_reg = 0.01 * (torch.mean((U-x)**2+(L-x)**2)) + 10 * F.relu(L-U).mean()
    
    return loss + loss_reg

def make_tif(name, data, output_path=""):   
    left=-180
    right=180
    up=90
    down=-90               
    dpi=0.05 

    geotransform = (left,dpi,0,up,0,dpi)
    spei_ds = gdal.GetDriverByName('Gtiff').Create(output_path+name+".tif",int((right-left)/dpi),int((up-down)/dpi), 1, gdal.GDT_Float32)
    spei_ds.SetGeoTransform(geotransform)
  
    srs = osr.SpatialReference() 
    srs.ImportFromEPSG(4326) 
    spei_ds.SetProjection(srs.ExportToWkt())  

    spei_ds.GetRasterBand(1).WriteArray(np.array(data)) 
    spei_ds.FlushCache() 
    spei_ds = None  
    print('Succeeded in generating {}.tif'.format(name),":)")
    
def get_month_img(DEM_path, VCD_path):
    
    h = np.zeros(25920000)
    alt = np.array(pi.open(DEM_path).getdata())

    raw_img = np.array([alt,h]).transpose()

    column = np.array(pi.open(VCD_path).getdata())
    zero_tag = (column<=0)  
    log_column = np.log(column+zero_tag)
    
    return np.hstack([log_column.reshape(-1,1),raw_img]), zero_tag
