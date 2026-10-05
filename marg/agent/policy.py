from collections.abc import Sequence
from typing import Literal

from marg.vision.models import SurveyInstance


def escalation_priority(
    confirmed_instances: Sequence[SurveyInstance],
) -> Literal["medium", "high"] | None:
    """Return the approval priority for confirmed, actionable segment evidence."""
    if any(item.severity >= 5 for item in confirmed_instances):
        return "high"
    if any(item.severity >= 4 for item in confirmed_instances) or sum(
        item.class_name == "D40" for item in confirmed_instances
    ) >= 3:
        return "medium"
    return None
