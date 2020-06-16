import numpy as np
import traittypes
import traitlets


def check_shape(*dimensions):
    def validator(trait, value):
        if len(value.shape) != len(dimensions):
            raise traitlets.TraitError(
                f"Expected rank {len(dimensions)} but got " f"rank {len(value.shape)}"
            )
        for a, b in zip(value.shape, dimensions):
            if b is not None and a != b:
                raise traitlets.TraitError(
                    f"Expected shape of {dimensions} but got " f"shape of {value.shape}"
                )
        return value

    return validator


def check_dtype(dtype):
    def validator(trait, value):
        if value.dtype != dtype:
            raise traitlets.TraitError(
                f"Expected dtype {dtype} but got " f"{value.dtype}"
            )
        return value

    return validator
