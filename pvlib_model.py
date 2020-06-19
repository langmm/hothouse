import pvlib


def sun_model(latitude, longitude, date, altitude=10.0, **kwargs):
    date = pvlib.tools._datetimelike_scalar_to_datetimeindex(date)
    solpos = pvlib.solarposition.get_solarposition(
        date, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(date)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    tl = pvlib.clearsky.lookup_linke_turbidity(date, latitude, longitude)
    cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
                                 dni_extra=dni_extra, altitude=altitude)
    # Convert from irradiance to PPFD
    eta_par = 0.368
    eta_photon = 4.56  # µmol s−1 W−1
    irr2ppfd = eta_par * eta_photon
    ppfd = {'direct': irr2ppfd * cs['dni'][0],
            'diffuse': irr2ppfd * cs['dhi'][0]}
    return ppfd
