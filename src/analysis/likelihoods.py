"""
Likelihood functions for dark-matter recoil analyses.

This module provides generic unbinned extended-likelihood
machinery together with a dark-matter-specific likelihood
based on a RateCalculator prediction.

Classes
-------
ExtendedUnbinnedLikelihood
    Generic extended likelihood consisting of a Poisson
    counting term and an unbinned shape term.

DMRecoilExtendedLikelihood
    Extended likelihood for recoil-energy data using
    predictions from a RateCalculator.

Notes
-----
The extended likelihood is

    log L = -mu + N log(mu) - log(N!)
            + sum_i log f(x_i)

where mu is the expected number of events and f(x)
is a normalized probability-density function.
"""


from src.rates import RateCalculator
from src.halo.shm import StandardHaloModel, IsotropicSpeedDistribution
from src.interactions.particles import WIMP
from src.detectors.detectors import BaseDetector
from src.utils.factory import create_interaction_si

from collections.abc import Callable
from scipy.special import gammaln
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
import numpy as np

class ExtendedUnbinnedLikelihood:
    """
    Generic unbinned extended likelihood.

    The likelihood combines a Poisson counting term for the
    total number of observed events with an unbinned PDF
    describing the distribution of event observables.

    Parameters
    ----------
    mu : float
        Expected number of events.
    f : callable or None
        Normalized probability-density function evaluated
        as f(x). Must return strictly positive values
        inside the support of the model.
        If None, the PDF contribution is ignored.

    Notes
    -----
    The log-likelihood is

        log L = -mu + N log(mu) - log(N!)
            + sum_i log f(x_i)

    where N is the number of observed events.
    """
    def __init__(self,
                 mu: float,
                 f: Callable[[ArrayLike], float] | None):

        if not np.isfinite(mu) or mu < 0:
            raise ValueError("Likelihood parameter mu must be finite and non-negative.")
        self.mu = mu

        if f is not None and not callable(f):
             raise ValueError("Likelihood parameter f must be callable.")
        self.f = f

    def logL_poisson(self,
                     N: int) -> float:
        """
        Return the Poisson log-likelihood.

        Parameters
        ----------
        N : int
            Number of observed events.

        Returns
        -------
        float
            Logarithm of the Poisson probability
            P(N | mu).
        """
        if N < 0:
            raise ValueError("Likelihood data N must be non-negative.")

        if self.mu == 0:
            return 0.0 if N == 0 else -np.inf

        return -self.mu + N * np.log(self.mu) - gammaln(N+1)

    def logL_poisson_max(self,
                         N: int) -> float:
        """
        Return the Poisson log-likelihood evaluated at the
        maximum-likelihood estimate mu_hat = N.

        Parameters
        ----------
        N : int
            Number of observed events.

        Returns
        -------
        float
            Maximum achievable Poisson log-likelihood for the
            specified event count.
        """
        if N < 0:
            raise ValueError("Likelihood data N must be non-negative.")

        if N == 0:
            return 0.

        return N * (np.log(N) - 1) - gammaln(N+1)

    def logL_pdf(self,
                 x: ArrayLike) -> float:
        """
        Return the unbinned PDF contribution.

        Parameters
        ----------
        x : ArrayLike
            Observed values of the analysis variable.

        Returns
        -------
        float
            Sum of logarithms of the probability density
            evaluated at each observation.

        Notes
        -----
        Returns -np.inf if any PDF value is non-positive.
        """
        if self.f is None:
            return 0.

        xi = np.asarray(x)

        if xi.ndim != 1:
            raise ValueError("Observations must be a one-dimensional array.")

        if not np.all(np.isfinite(xi)):
            raise ValueError("Observations must contain only finite values.")

        pdf_xi = np.asarray(self.f(xi), dtype=float)

        if pdf_xi.shape != xi.shape:
            raise ValueError("PDF must return one value per observation.")

        if not np.all(np.isfinite(pdf_xi)):
            return -np.inf

        if np.any(pdf_xi <= 0):
            return -np.inf

        return np.sum(np.log(pdf_xi))

    def logL(self,
             x: ArrayLike) -> float:
        """
        Return the full extended log-likelihood.

        Parameters
        ----------
        x : ArrayLike
            Observed events.

        Returns
        -------
        float
            Extended log-likelihood combining counting and
            shape information.
        """

        xi = np.asarray(x)
        N = len(xi)
        return self.logL_poisson(N=N) + self.logL_pdf(x=xi)

    def logL_muhat(self,
                   x: ArrayLike) -> float:

        """
        Return the profile likelihood with respect to mu.

        The Poisson term is evaluated at its maximum-likelihood
        estimate mu_hat = N while the PDF contribution is left
        unchanged.

        Parameters
        ----------
        x : ArrayLike
            Observed events.

        Returns
        -------
        float
            Log-likelihood evaluated at mu_hat.
        """

        xi = np.asarray(x)
        N = len(xi)

        return self.logL_poisson_max(N=N) + self.logL_pdf(x=xi)
        

class DMRecoilExtendedLikelihood(ExtendedUnbinnedLikelihood):
    """
    Extended likelihood for dark-matter recoil-energy data.

    This class constructs an extended likelihood directly
    from a RateCalculator instance.
    The expected number of events and normalized recoil spectrum
    are obtained from the supplied rate calculator.

    Parameters
    ----------
    rate_calculator : RateCalculator
        Object providing expected_counts() and
        interpolate_normalized_spectrum().
    """

    def __init__(self, rate_calculator: RateCalculator):

        if not isinstance(rate_calculator, RateCalculator):
            raise TypeError("Input rate_calculator is not a RateCalculator.")

        mu = rate_calculator.expected_counts()
        f = rate_calculator.interpolate_normalized_spectrum

        super().__init__(mu, f)

#------------
# SI specific
#------------

@dataclass
class LikelihoodGridVSMassDelta:
    """
    Likelihood on a two-dimensional
    grid of WIMP masses and mass splittings.

    Attributes
    ----------
    masses : NDArray
        WIMP masses in eV.

    deltas : NDArray
        Mass splittings in eV.

    logL : NDArray
        Profile log likelihood evaluated at μ̂=N
        and shifted so that the global maximum
        satisfies max(logL) = 0.
        Consequently logL contains ΔlogL rather
        than the absolute log likelihood.

    """
    masses: NDArray
    deltas: NDArray
    logL: NDArray

    @property
    def q(self):
        """
        Likelihood-ratio test statistic.

        Returns
        -------
        NDArray
            Test statistic q = -2 ΔlogL,
            where ΔlogL is the log likelihood
            shifted so that the global maximum
            satisfies ΔlogL = 0.

        The best-fit point therefore has q = 0 and larger
        values indicate poorer agreement with the data.
        """
        return - 2 * self.logL

    @property
    def q_cap_25(self):
        """
        Likelihood-ratio test statistic
        capped at q=25
        """

        return np.where(np.isfinite(self.q),
                      np.minimum(self.q, 25), np.inf)

    @property
    def best_fit_index(self):
        """
        Index of the best-fit grid point.

        Returns
        -------
        tuple[int, int]
            Indices (i, j) corresponding to the minimum
            value of q, equivalently the maximum likelihood.
        """
        return np.unravel_index(
            np.argmin(self.q),
            self.q.shape)

    @property
    def best_fit_mass(self):
        """
        Best-fit WIMP mass in eV.

        Returns
        -------
        float
            WIMP mass (in eV) corresponding to the maximum
            likelihood point on the grid.
        """
        imin, _ = self.best_fit_index
        return self.masses[imin]

    @property
    def best_fit_delta(self):
        """
        Best-fit delta (inelastic mass splitting) in eV.

        Returns
        -------
        float
            Mass splitting (in eV) corresponding to the maximum
            likelihood point on the grid.
        """
        _, jmin = self.best_fit_index
        return self.deltas[jmin]

    @property
    def meshgrid(self):
        """
        Meshgrid representation of masses and deltas
        suitable for contour and density plots.
        """
        return np.meshgrid(
            np.asarray(self.masses),
            np.asarray(self.deltas),
            indexing="ij",
        )

    @property
    def shape(self):
        """
        Shape of the likelihood grid.

        Returns
        -------
        tuple[int, int]
            Number of scanned masses and mass splittings.
        """
        return self.logL.shape


def likelihood_grid_si_mass_delta(
    wimp_masses: ArrayLike,
    deltas: ArrayLike,
    detector: BaseDetector,
    energies: NDArray,
    halo: IsotropicSpeedDistribution | None = None,
    **interaction_kwargs,
) -> LikelihoodGridVSMassDelta:
    """
    Compute likelihood over a grid of WIMP masses
    and mass splittings for spin-independent
    scattering.

    For each point (mX, delta), a spin-independent
    interaction model is constructed for the given detector.

    Parameters
    ----------
    wimp_masses : ArrayLike
        WIMP masses in eV.

    deltas : ArrayLike
        Mass splittings in eV.

    detector : BaseDetector
        Detector for which the matched signal
        prediction is computed.

    energies : NDarray,
        Event energies in eV.

    halo : IsotropicSpeedDistribution, optional
        Halo model used for both detectors.
        If None, the Standard Halo Model is used.

    Returns
    -------
    LikelihoodGrid
        Container holding the scanned masses,
        mass splittings, and the corresponding
        log likelihood.

    Notes
    -----
    The likelihood at each grid point is evaluated
    using DMRecoilExtendedLikelihood.

    The expected event count is profiled out by
    evaluating the likelihood at μ̂=N, where N is
    the number of observed events.

    Consequently the scan is sensitive primarily
    to spectral shape differences between models
    rather than to the assumed interaction
    normalization.

    The returned array ``logL[i, j]``
    corresponds to

    - ``masses[i]``
    - ``deltas[j]``
    """

    interaction_kwargs = (
        {} if interaction_kwargs is None
        else interaction_kwargs)

    if halo is None:
        halo = StandardHaloModel().boost()

    wimp_masses = np.asarray(wimp_masses)
    deltas = np.asarray(deltas)

    nucleus = detector.nucleus

    result = np.empty((len(wimp_masses), len(deltas)), dtype=float)

    if result.size == 0:
        raise ValueError("Likelihood grid with zero size.")

    logLmax = -np.inf

    for i, mX in enumerate(wimp_masses):

        wimp = WIMP(mass=mX, spin=0)

        for j, delta in enumerate(deltas):
        
            si = create_interaction_si(wimp=wimp,
                                       nucleus=nucleus,
                                       delta=delta,
                                       **interaction_kwargs)
        
            rc = RateCalculator(halo=halo,
                                interaction=si,
                                detector=detector)

            llh = DMRecoilExtendedLikelihood(rate_calculator=rc)
            logL = llh.logL_muhat(x=energies)

            result[i, j] = logL
            logLmax = max(logLmax, logL)

    if np.isposinf(logLmax):
        raise RuntimeError("Likelihood maximum is infinity.")

    result -= logLmax

    if not np.any(np.isfinite(result)):
        RuntimeWarning("No scanned model has a finite likelihood.")

    return LikelihoodGridVSMassDelta(
        masses=np.asarray(wimp_masses),
        deltas=np.asarray(deltas),
        logL = result,
    )