"""TrustCurve — model-agnostic interpretability plots.

Polars-first PDP, ICE, and ALE for any predict function.

    >>> import trustcurve as tc
    >>> result = tc.pdp(model.predict, X, "age")
    >>> result.plot()

"""

from trustcurve._compute import ale, ice, interaction, pdp
from trustcurve._types import ALEResult, ICEResult, InteractionResult, PDPResult

__all__ = [
    "pdp",
    "ice",
    "ale",
    "interaction",
    "PDPResult",
    "ICEResult",
    "ALEResult",
    "InteractionResult",
]
