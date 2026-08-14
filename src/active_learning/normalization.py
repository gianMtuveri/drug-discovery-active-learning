from collections.abc import Callable

import numpy as np


ScoreArray = np.ndarray
NormalizationFunction = Callable[[ScoreArray], ScoreArray]


def identity_scale(values: ScoreArray) -> ScoreArray:
    """
    Return scores without normalization.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "identity_scale expects a one-dimensional array."
        )

    return values.copy()


def robust_scale(values: ScoreArray) -> ScoreArray:
    """
    Normalize a one-dimensional array using its median and IQR.

    The transformation is:

        scaled = (values - median) / IQR

    If the IQR is zero, the standard deviation is used as a fallback.
    If both are zero, an array of zeros is returned.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "robust_scale expects a one-dimensional array."
        )

    if len(values) == 0:
        return values.copy()

    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)

    if np.isclose(iqr, 0.0):
        scale = float(np.std(values))

        if np.isclose(scale, 0.0):
            return np.zeros_like(
                values,
                dtype=float,
            )
    else:
        scale = iqr

    return (values - median) / scale


NORMALIZATION_FUNCTIONS: dict[
    str,
    NormalizationFunction,
] = {
    "none": identity_scale,
    "robust": robust_scale,
}


def normalize_scores(
    values: ScoreArray,
    method: str,
) -> ScoreArray:
    """
    Normalize scores using a registered normalization method.
    """
    try:
        normalization_function = (
            NORMALIZATION_FUNCTIONS[method]
        )
    except KeyError as error:
        valid_methods = ", ".join(
            sorted(NORMALIZATION_FUNCTIONS)
        )

        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Available methods: {valid_methods}"
        ) from error

    return normalization_function(values)