import math

def solarFParallel(lat, lon, timezone_lon, day, hour):
    """ Calculate solar elevation, zenith and azimuth angles
    
    Inputs:
        
        lat             ::  latitude (decimal degree)
        lon             ::  longitude (decimal degree)
        timezone_lon    ::  longitude of standard meridian (decimal degree)
        doy             ::  day of the year (1-366)
        hour            ::  hour of the day (decimal, e.g. 12:30 = 12.5
        
    Outputs:
        
        beta            ::  solar elevation angle (radians)
        zeni            ::  zenith angle (radians)
        azi             ::  solar azimuth angle (radians)
    """

    # Convert degree to radians
    FAC = math.pi / 180.0

    # Solar declinations (radians)
    dec = math.asin(0.39785 * math.sin((278.97 + 0.9856 * day + 1.9165 *
        math.sin((356.6 + 0.9856 * day) * FAC)) * FAC))

    # Day length in hours
    length = math.acos(-1.0 * (math.sin(lat * FAC) * math.sin(dec)) /
        (math.cos(lat * FAC) * math.cos(dec))) / FAC * 2.0/15.0

    # Teta (radians), time equation (hours)
    teta = (279.575 + 0.9856 * day) * FAC
    timeEq = (-104.7 * math.sin(teta) + 596.2 * math.sin(2.0 * teta) + 4.3 *
        math.sin(3.0 * teta) - 12.7 * math.sin(4.0 * teta) - 429.3 *
        math.cos(teta) - 2.0 * math.cos(2.0 * teta) + 19.3 * math.cos(3.0 * teta)) / 3600.0

    # Longitude correction (hours)
    LC = (timezone_lon - lon) / 15.0

    # Solar noon (hours) / solar time (hours)
    solarnoon = 12.0 - LC - timeEq 
    solartime = hour - LC - timeEq

    # Solar elevation
    beta = math.asin(math.sin(lat * FAC) * math.sin(dec) + math.cos(lat * FAC) * 
        math.cos(dec) * math.cos(15.0 * FAC * (solartime-solarnoon)))

    # Zenith angle (radians)
    zeni = math.pi/2.0 - beta
    
    # Azimuth angle (radians)
    azi = math.acos((math.sin(lat * FAC) * math.cos(zeni) - math.sin(dec)))/math.cos(lat*FAC)*math.sin(zeni)
    
    if (solartime < solarnoon):
        azi = azi * -1.0

    So = 1367.0 * (1 + 0.033 * math.cos(2.0 * math.pi * day / 366.0)) * math.cos(zeni)

    return beta, zeni, azi, So