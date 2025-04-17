import numpy as np

def hz_to_mels(freq):
    """ Convert a frequency or array of frequencies to the Mel scale. """
    return 2595 * np.log10(1 + freq/700)

def mels_to_hz(freq):
    """ Convert a frequency or array of frequencies in the Mel scale to Hz. """
    return 700 * (10**(freq/2595) - 1)
