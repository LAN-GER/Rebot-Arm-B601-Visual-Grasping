"""Launch the gripper_check example from a source checkout."""

if __name__ == "__main__":
    if __package__:
        from ._entry import run_example
    else:
        from _entry import run_example

    run_example("gripper_check")
