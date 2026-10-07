"""
Poisson-based statistical methods for counting experiments.

The classes in this module provide confidence limits based on
Poisson counting statistics. The current implementation supports
the simple case of zero observed events and zero expected
background, commonly used for sensitivity estimates in rare-event
searches.

Future versions may support non-zero background expectations,
confidence intervals, and additional statistical methods.
"""

from math import log

from src.sensitivity.types import CountingExperiment


class PoissonUpperLimit:
    """
    Upper limit on the expected signal count based on Poisson
    statistics.

    Parameters
    ----------
    cl :
        Confidence level expressed as a number between 0 and 1.

    Notes
    -----
    The current implementation supports only the special case of
    zero observed events and zero expected background.

    For this case the upper limit is

    .. math::

        s_{up} = -\\ln(1 - CL)

    where ``CL`` is the chosen confidence level.
    """

    def __init__(self,
                 cl: float = 0.9):

        self.cl = cl

        if not (0.0 < self.cl < 1.0):
            raise ValueError("Confidence level must satisfy 0 < cl < 1.")

    def signal_upper_bound(self,
                           experiment: CountingExperiment
                           ) -> float:
        """
        Compute the upper bound on the expected signal count.

        Parameters
        ----------
        experiment : CountingExperiment
            Counting experiment definition.

        Returns
        -------
        float
            Upper limit on the expected number of signal events.

        Raises
        ------
        NotImplementedError
        If the experiment does not satisfy
        ``counts == 0`` and ``background == 0``.

        Notes
        -----
        For zero observed events and zero background expectation,
        the upper bound is computed analytically as

        .. math::

        s_{up} = -\\ln(1 - CL).
        """
        
        if experiment.counts != 0:
            raise NotImplementedError("Only counts = 0 case implemented")
        if experiment.background != 0.0:
            raise NotImplementedError("Only background = 0 case implemented")

        return - log(1 - self.cl)
