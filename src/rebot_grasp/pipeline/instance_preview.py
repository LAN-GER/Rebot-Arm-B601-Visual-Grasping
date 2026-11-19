"""Run independent AnyGrasp preview queries for the detected instances of one frame."""
from dataclasses import dataclass, replace

from rebot_grasp.core.types import GraspCandidate, Target
from rebot_grasp.perception.target_selection import select_target
from rebot_grasp.pipeline.planning import plan_capture


@dataclass(frozen=True)
class InstanceGrasp:
    target: Target
    filtered: GraspCandidate | None
    raw: GraspCandidate | None
    status: str


def plan_instance_previews(capture, targets, backend, config):
    selected = select_target(targets, config.target_class)
    instances = []

    def record(target, result):
        instances.append(InstanceGrasp(target, next(iter(result.candidates), None),
                                       getattr(backend, "reference", None), result.status))

    for target in targets:
        if target is selected or not target.mask.any():
            continue
        result = plan_capture(capture, [target], backend, replace(config, target_class=target.label))
        record(target, result)

    # Leave the active class's diagnostics in the backend for the 3-D and depth windows.
    if hasattr(backend, "reset_diagnostics"):
        backend.reset_diagnostics()
    result = plan_capture(capture, targets, backend, config)
    if selected is not None:
        record(selected, result)
    return result, selected, instances
