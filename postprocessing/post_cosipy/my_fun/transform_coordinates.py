import numpy as np

def transform_coordinates(coords):
    """ Transform coordinates from geodetic to cartesian
    an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared    
    
    coords = np.asarray(coords).astype(np.float)
                                                      
    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])
    
    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1]) 
    
    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    
    return np.column_stack((x, y, z))


