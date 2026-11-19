from rebot_grasp.core.types import Target


def select_target(targets: list[Target], label: str | None = None) -> Target | None:
    choices = [target for target in targets if target.mask.any() and (label is None or target.label.casefold() == label.casefold())]
    return max(choices, key=lambda target: target.confidence) if choices else None
