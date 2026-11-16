import numpy as np
import pytest
from scipy.special import gammaln

from src.analysis.likelihoods import ExtendedUnbinnedLikelihood

def constant_pdf(x):
    return np.ones_like(np.asarray(x), dtype=float)


def gaussian_pdf(x):
    x = np.asarray(x)
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class BadPDFNegative:
    def __call__(self, x):
        return -np.ones_like(np.asarray(x))


class BadPDFZero:
    def __call__(self, x):
        return np.zeros_like(np.asarray(x))


class BadPDFNan:
    def __call__(self, x):
        x = np.asarray(x)
        return np.full_like(x, np.nan, dtype=float)


# -------------------------
# constructor tests
# -------------------------

def test_negative_mu_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ExtendedUnbinnedLikelihood(mu=-1.0, f=constant_pdf)


def test_nonfinite_mu_raises():
    with pytest.raises(ValueError, match="finite"):
        ExtendedUnbinnedLikelihood(mu=np.inf, f=constant_pdf)


def test_noncallable_pdf_raises():
    with pytest.raises(ValueError, match="callable"):
        ExtendedUnbinnedLikelihood(mu=1.0, f=42)


# -------------------------
# poisson term
# -------------------------

def test_logL_poisson():
    llh = ExtendedUnbinnedLikelihood(mu=10.0, f=None)
    n = 3
    expected = (
        -10.0
        + n * np.log(10.0)
        - gammaln(n+1)
    )

    assert np.isclose(llh.logL_poisson(N=n), expected)


def test_logL_poisson_negative_N():
    llh = ExtendedUnbinnedLikelihood(mu=1.0, f=None)

    with pytest.raises(ValueError):
        llh.logL_poisson(-1)


def test_logL_poisson_max():
    llh = ExtendedUnbinnedLikelihood(mu=5.0, f=None)

    n= 7
    expected = (
        n * (np.log(n) - 1)
        - gammaln(n + 1)
    )

    assert np.isclose(
        llh.logL_poisson_max(N=n),
        expected
    )


def test_logL_poisson_max_N0():
    llh = ExtendedUnbinnedLikelihood(mu=5.0, f=None)

    assert llh.logL_poisson_max(0) == 0.0


# -------------------------
# pdf term
# -------------------------

def test_logL_pdf_constant():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=constant_pdf
    )

    x = np.array([1, 2, 3])

    assert llh.logL_pdf(x=x) == 0.0


def test_logL_pdf_gaussian():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=gaussian_pdf
    )

    x = np.array([0.0])

    expected = np.log(
        1.0 / np.sqrt(2 * np.pi)
    )

    assert np.isclose(
        llh.logL_pdf(x=x),
        expected
    )


def test_logL_pdf_negative_returns_minus_inf():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=BadPDFNegative()
    )

    assert llh.logL_pdf([1]) == -np.inf


def test_logL_pdf_zero_returns_minus_inf():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=BadPDFZero()
    )

    assert llh.logL_pdf([1]) == -np.inf


def test_logL_pdf_none_returns_zero():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=None
    )

    assert llh.logL_pdf([1, 2, 3]) == 0.0


# -------------------------
# full likelihood
# -------------------------

def test_logL_equals_sum_of_components():
    llh = ExtendedUnbinnedLikelihood(
        mu=5.0,
        f=constant_pdf
    )

    x = np.array([1.0, 2.0])

    expected = (
        llh.logL_poisson(2)
        + llh.logL_pdf(x)
    )

    assert np.isclose(
        llh.logL(x),
        expected
    )


def test_logL_muhat_equals_sum_of_components():
    llh = ExtendedUnbinnedLikelihood(
        mu=5.0,
        f=constant_pdf
    )

    x = np.array([1.0, 2.0, 3.0])

    expected = (
        llh.logL_poisson_max(3)
        + llh.logL_pdf(x)
    )

    assert np.isclose(
        llh.logL_muhat(x),
        expected
    )


# -------------------------
# edge cases
# -------------------------

def test_empty_dataset():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=constant_pdf
    )

    assert np.isfinite(llh.logL([]))


def test_single_event_dataset():
    llh = ExtendedUnbinnedLikelihood(
        mu=1.0,
        f=constant_pdf
    )

    assert np.isfinite(llh.logL([0.5]))
