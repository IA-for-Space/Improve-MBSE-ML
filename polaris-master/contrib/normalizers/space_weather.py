"""
Normalizer class for space weather data

Adds units of each data item
"""
from contrib.normalizers.common import Field, Normalizer


# Geomagnetic Data
class DGD(Normalizer):
    """
    The class provides equations for normalization of Geomagnetic Data
    from NOAA
    """
    def __init__(self):
        super(DGD, self).__init__()
        self.normalizers = [
            Field('Fredericksburg A', lambda x: x, None, 'Fredericksburg A'),
            Field('Fredericksburg K 0-3', lambda x: x, None,
                  'Fredericksburg K 0-3'),
            Field('Fredericksburg K 3-6', lambda x: x, None,
                  'Fredericksburg K 3-6'),
            Field('Fredericksburg K 6-9', lambda x: x, None,
                  'Fredericksburg K 6-9'),
            Field('Fredericksburg K 9-12', lambda x: x, None,
                  'Fredericksburg K 9-12'),
            Field('Fredericksburg K 12-15', lambda x: x, None,
                  'Fredericksburg K 12-15'),
            Field('Fredericksburg K 15-18', lambda x: x, None,
                  'Fredericksburg K 15-18'),
            Field('Fredericksburg K 18-21', lambda x: x, None,
                  'Fredericksburg K 18-21'),
            Field('Fredericksburg K 21-24', lambda x: x, None,
                  'Fredericksburg K 21-24'),
            Field('College A', lambda x: x, None, 'College A'),
            Field('College K 0-3', lambda x: x, None, 'College K 0-3'),
            Field('College K 3-6', lambda x: x, None, 'College K 3-6'),
            Field('College K 6-9', lambda x: x, None, 'College K 6-9'),
            Field('College K 9-12', lambda x: x, None, 'College K 9-12'),
            Field('College K 12-15', lambda x: x, None, 'College K 12-15'),
            Field('College K 15-18', lambda x: x, None, 'College K 15-18'),
            Field('College K 18-21', lambda x: x, None, 'College K 18-21'),
            Field('College K 21-24', lambda x: x, None, 'College K 21-24'),
            Field('Planetary A', lambda x: x, None, 'Planetary A'),
            Field('Planetary K 0-3', lambda x: x, None, 'Planetary K 0-3'),
            Field('Planetary K 3-6', lambda x: x, None, 'Planetary K 3-6'),
            Field('Planetary K 6-9', lambda x: x, None, 'Planetary K 6-9'),
            Field('Planetary K 9-12', lambda x: x, None, 'Planetary K 9-12'),
            Field('Planetary K 12-15', lambda x: x, None, 'Planetary K 12-15'),
            Field('Planetary K 15-18', lambda x: x, None, 'Planetary K 15-18'),
            Field('Planetary K 18-21', lambda x: x, None, 'Planetary K 18-21'),
            Field('Planetary K 21-24', lambda x: x, None, 'Planetary K 21-24'),
        ]


# Particle data
class DPD(Normalizer):
    """
    The class provides equations for normalization of Particle Data from NOAA
    """
    def __init__(self):
        super(DPD, self).__init__()
        self.normalizers = [
            Field('Proton 1 MeV', lambda x: x, 'Protons/(cm^2*day*sr)',
                  'Proton Flux >1 MeV'),
            Field('Proton 10 MeV', lambda x: x, 'Protons/(cm^2*day*sr)',
                  'Proton Flux >10 MeV'),
            Field('Proton 100 MeV', lambda x: x, 'Protons/(cm^2*day*sr)',
                  'Proton Flux >100 MeV'),
            Field('Electron 800 KeV', lambda x: x, 'Electrons/(cm^2*day*sr)',
                  'Electron Flux >800 KeV'),
            Field('Electron 2 MeV', lambda x: x, 'Electrons/(cm^2*day*sr)',
                  'Electron Flux >2 MeV'),
            Field('Neutron', lambda x: x, '%',
                  'Neutron Monitor as % of background'),
        ]


# Solar Data
class DSD(Normalizer):
    """
    The class provides equations for normalization of Solar Data from NOAA
    """
    def __init__(self):
        super(DSD, self).__init__()
        self.normalizers = [
            # Solar Flux Units conversion
            # https://www.researchgate.net/figure/Solar-radio-flux-in-Solar-Flux-Units-1-SFU-10-22-W-m-2-Hz-1-measured-in-Humain-on_fig5_318360593
            # Stankov, Stanimir & Bergeot, Nicolas & Berghmans, David &
            # Bolsée, David & Bruyninx, Carine & Chevalier, Jean-Marie &
            # Clette, Frédéric & De Backer, Hugo & Keyser, Johan & D’Huys,
            # Elke & Dominique, Marie & Lemaire, Joseph & Magdalenic, Jasmina
            # & Marqué, Christophe & Pereira, Nuno & Pierrard, Viviane &
            # Sapundjiev, Danislav & Seaton, D. & Stegen, Koen & West, Matthew.
            # (2017). Multi-instrument observations of the solar eclipse on 20
            # March 2015 and its effects on the ionosphere over Belgium and
            # Europe. Journal of Space Weather and Space Climate.
            # 7. A19. 10.1051/swsc/2017017.
            Field('Radio Flux', lambda x: x * 1e-22, 'W / (m^2 * Hz)',
                  'Radio Flux 10.7cm (2800 MHz)'),
            Field('SESC sunspot number', lambda x: x, None,
                  'SESC sunspot number'),
            Field('Sunspot Area', lambda x: x * 10e-6, 'Solar Hemispheres',
                  'Sunspot Area'),
            Field('New Regions', lambda x: x, None, 'New Regions'),
            Field('Solar Mean Field', lambda x: x * 1e-4, 'kg / (A * s^2)',
                  'Solar Mean Magnetic Field'),
            Field('X-ray Background Flux', _x_ray_convert, 'W / m^2',
                  'GOES X-Ray Background Flux'),
            Field('X-Ray C', lambda x: x, None,
                  'Number of flares in X-Ray C class'),
            Field('X-Ray M', lambda x: x, None,
                  'Number of flares in X-Ray M class'),
            Field('X-Ray X', lambda x: x, None,
                  'Number of flares in X-Ray X class'),
            Field('X-Ray S', lambda x: x, None,
                  'Number of flares in X-Ray S class'),
            Field('Optical 1', lambda x: x, None,
                  'Number of Optical 1 flares'),
            Field('Optical 2', lambda x: x, None,
                  'Number of Optical 2 flares'),
            Field('Optical 3', lambda x: x, None,
                  'Number of Optical 3 flares'),
        ]


def _x_ray_convert(flux_index):

    if flux_index == -1:
        return flux_index

    x_ray_class = flux_index[0]

    if x_ray_class == 'A':
        scaler = 1e-8
    elif x_ray_class == 'B':
        scaler = 1e-7
    elif x_ray_class == 'C':
        scaler = 1e-6
    elif x_ray_class == 'M':
        scaler = 1e-5
    elif x_ray_class == 'X':
        scaler = 1e-4
    else:
        raise ValueError("Value {} is not understood for x_ray_flux")

    return float(flux_index[1:]) * scaler
