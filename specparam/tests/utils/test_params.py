"""Test functions for specparam.utils.params."""

from specparam.utils.params import *

###################################################################################################
###################################################################################################

def test_compute_knee_frequency():

    assert compute_knee_frequency(100, 2)

def test_compute_time_constant():

    assert compute_time_constant(10)

def test_compute_fwhm():

    assert compute_fwhm(1.5)

def test_compute_gauss_std():

    assert compute_gauss_std(1.0)
