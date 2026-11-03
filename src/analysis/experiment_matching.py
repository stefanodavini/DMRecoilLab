"""
Utilities for matching signal hypotheses between
direct-detection experiments.

The functions in this module infer the interaction
cross section required to explain a signal observed
in one detector and predict the corresponding signal
expected in a second detector.

When the two detectors use different target nuclei,
an interaction model appropriate for the second
target must be supplied explicitly.

The prediction assumes that event rates scale
linearly with the interaction-model reference
cross section.
"""

import numpy as np

from src.rates import RateCalculator
from src.detectors.detectors import BaseDetector
from src.interactions.base_interaction_model import InteractionModel

def predict_counts_from_counts_match(
    rate_calculator_a: RateCalculator,
    counts_a: float,
    detector_b: BaseDetector,
    interaction_b: InteractionModel | None = None,
) -> float:
    """
    Predict the number of signal counts expected in
    detector B under the hypothesis that the signal
    observed in detector A is produced by the same
    dark-matter model.

    Parameters
    ----------
    rate_calculator_a : RateCalculator
        Calculator describing detector A and the
        reference interaction model.

    counts_a : float
        Number of observed or hypothesized signal
        counts in detector A.

    detector_b : BaseDetector
        Detector for which the corresponding signal
        prediction should be computed.

    interaction_b : InteractionModel, optional
        Interaction model for detector B.
        Required when detector B uses
        a different target nucleus.

    Returns
    -------
    float
        Predicted number of signal counts in
        detector B.

    Raises
    ------
    ValueError
        If the detector and interaction targets are
        inconsistent or if a target change requires
        an interaction model that was not supplied.
        Also, if a required reference cross section 
        is non-positive.

    Notes
    -----
    The prediction is obtained by first computing the
    interaction-model reference cross section that
    reproduces the specified counts in detector A and
    then evaluating the corresponding expectation in
    detector B.
    """

    if rate_calculator_a.target_nucleus == detector_b.nucleus:
        return predict_counts_from_counts_match_same_target(
            rate_calculator_a=rate_calculator_a,
            counts_a=counts_a,
            detector_b=detector_b)

    else:
        if interaction_b is None:
            raise ValueError("interaction_b is required when the target "
                             "nucleus differs between detector A and detector B.")
        
        return predict_counts_from_counts_match_different_target(
            rate_calculator_a=rate_calculator_a,
            counts_a=counts_a,
            detector_b=detector_b,
            interaction_b = interaction_b)


def predict_counts_from_counts_match_same_target(
    rate_calculator_a: RateCalculator,
    counts_a: float,
    detector_b: BaseDetector) -> float:
    """
    Predict counts in detector B when both detectors
    use the same target nucleus.
    """

    if not rate_calculator_a.target_nucleus == detector_b.nucleus:
        raise ValueError("predict_counts_from_counts_match_same_target "
                         "requires identical target nuclei.")

    sigma_hat = rate_calculator_a.expected_cross_section_from_counts(counts=counts_a)

    rate_calculator_b = RateCalculator(
        halo=rate_calculator_a.halo,
        interaction=rate_calculator_a.interaction,
        detector=detector_b,
        rho_dm=rate_calculator_a.rho_dm)

    return _predict_counts(sigma_hat=sigma_hat,
                           rate_calculator=rate_calculator_b)


def predict_counts_from_counts_match_different_target(
    rate_calculator_a: RateCalculator,
    counts_a: float,
    detector_b: BaseDetector,
    interaction_b: InteractionModel) -> float:
    """
    Predict counts in detector B when detector B
    requires a different interaction model because
    the target nucleus differs from detector A.
    """

    if not interaction_b.nucleus == detector_b.nucleus:
        raise ValueError("interaction_b.nucleus and detector_b.nucleus "
                         "must be identical.")

    if interaction_b.mass_wimp != rate_calculator_a.mass_wimp:
        raise ValueError("interaction_b and rate_calculator_a must "
                         "describe the same WIMP mass.")

    rate_calculator_b = RateCalculator(
        halo=rate_calculator_a.halo,
        interaction=interaction_b,
        detector=detector_b,
        rho_dm=rate_calculator_a.rho_dm)

    sigma_hat = rate_calculator_a.expected_cross_section_from_counts(counts=counts_a)

    return _predict_counts(sigma_hat=sigma_hat,
                           rate_calculator=rate_calculator_b)


def _predict_counts(sigma_hat: float,
                    rate_calculator: RateCalculator) -> float:
    """
    Scale the reference prediction of a rate calculator
    to the expected cross section sigma_hat.
    """

    if np.isnan(sigma_hat):
        return np.nan

    sigma_ref = rate_calculator.sigma_reference

    if sigma_ref <= 0:
        raise ValueError("Reference cross section must be positive.")

    return sigma_hat / sigma_ref * rate_calculator.expected_counts()
