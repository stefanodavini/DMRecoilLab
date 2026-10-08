"""
Conversion utilities between statistical count limits and
dark-matter cross-section limits.

The statistical methods implemented in the sensitivity package
operate on expected signal counts. This module converts those
count limits into limits on the WIMP-proton cross section using
a RateCalculator instance.

The returned cross section corresponds to the interaction-model
parameter used to generate the expected counts. For standard
spin-independent interactions this is the WIMP-proton cross
section sigma_p, which is the quantity conventionally reported
by direct-detection experiments.

The conversion assumes that the expected number of signal
events scales linearly with the cross section parameter.
"""


from src.rates import RateCalculator

from src.sensitivity.poisson import PoissonUpperLimit
from src.sensitivity.types import CountingExperiment, MassSigmaSensitivityPoint

def sigma_upper_limit(
    statistic: PoissonUpperLimit,
    rate_calculator: RateCalculator,
    experiment: CountingExperiment | None = None,
    ) -> MassSigmaSensitivityPoint:
    """
    Compute the upper limit on the WIMP-proton cross section.

    Parameters
    ----------
    statistic : PoissonUpperLimit
        Statistical model used to compute the signal-count
        upper bound.

    rate_calculator: RateCalculator
        Rate calculator configured with a detector and a
        reference WIMP-proton cross section.

    experiment: CountingExperiment
        Experiment's counts and expected background.
        If none is provided, assumes the reference
        experiment corresponding to
        - counts = 0
        - background = 0.

    Returns
    -------
    MassSigmaSensitivityPoint
        Sensitivity result containing the WIMP mass and the
        corresponding WIMP-proton cross-section upper limit.

    Notes
    -----
    The calculation proceeds as follows:

    1. Compute the signal-count upper limit
       s_up
       from the statistical model.

    2. Compute the expected number of signal events
       N_ref
       for the reference cross section specified in the
       RateCalculator.

    3. Rescale the cross section according to
       sigma_up = sigma_ref * s_up / N_ref.

    The current implementation assumes the reference
    experiment corresponding to
    - counts = 0
    - background = 0.

    The conversion returns limits on the interaction-model
    normalization parameter exposed through
    ``interaction.sigma_proton``.

    For spin-independent interactions this corresponds
    to the WIMP-proton cross section conventionally
    exported by direct-detection experiments.
    """
    if experiment is None:
        experiment = CountingExperiment(counts=0,
                                        background=0.0)

    s_up = statistic.signal_upper_bound(experiment)

    n_ref = rate_calculator.expected_counts()

    if n_ref <= 0:
        raise ValueError("Expected counts must be positive.")

    sigma_up = rate_calculator.interaction.sigma_proton * s_up / n_ref

    return MassSigmaSensitivityPoint(
        mass_wimp=rate_calculator.wimp.mass,
        sigma_p_upper=sigma_up,
        cl=statistic.cl
    )

