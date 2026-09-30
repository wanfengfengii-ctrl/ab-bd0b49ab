"""Request/response schemas for the delay-plan compiler API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .solver import Anchor, Instance

MIN_TARGETS = 12
MAX_TARGETS = 48
MIN_ANCHORS = 2
MAX_ANCHORS = 8
# Input envelope: every delay/target/step magnitude must stay within
# +/-VALUE_LIMIT so the integer DP costs cannot overflow int32.
VALUE_LIMIT = 100_000
MAX_RAMPS_LIMIT = 47  # n - 1 for the largest supported plan


class AnchorIn(BaseModel):
    """One element that must be hit exactly."""

    element: int = Field(ge=0, description="zero-based element index")
    delay: int = Field(ge=-VALUE_LIMIT, le=VALUE_LIMIT,
                       description="required exact delay at this element")


class CompileRequest(BaseModel):
    targets: list[int] = Field(
        min_length=MIN_TARGETS, max_length=MAX_TARGETS,
        description="per-element integer target delays")
    minDelay: int = Field(ge=-VALUE_LIMIT, le=VALUE_LIMIT,
                          description="lower bound of the global delay interval")
    maxDelay: int = Field(ge=-VALUE_LIMIT, le=VALUE_LIMIT,
                          description="upper bound of the global delay interval")
    maxStep: int = Field(ge=0, le=VALUE_LIMIT,
                         description="max allowed absolute change between adjacent elements")
    maxRamps: int = Field(ge=1, le=MAX_RAMPS_LIMIT,
                          description="max number of ramps (runs of equal adjacent differences)")
    anchors: list[AnchorIn] = Field(
        min_length=MIN_ANCHORS, max_length=MAX_ANCHORS,
        description="elements whose delays must be hit exactly")

    @model_validator(mode="after")
    def _cross_check(self) -> "CompileRequest":
        if self.minDelay > self.maxDelay:
            raise ValueError("minDelay must be <= maxDelay")
        n = len(self.targets)
        seen: set[int] = set()
        for a in self.anchors:
            if a.element >= n:
                raise ValueError(
                    f"anchor element {a.element} out of range for {n} targets")
            if a.element in seen:
                raise ValueError(f"duplicate anchor element {a.element}")
            seen.add(a.element)
        for i, t in enumerate(self.targets):
            if abs(t) > VALUE_LIMIT:
                raise ValueError(
                    f"targets[{i}]={t} exceeds the +/-{VALUE_LIMIT} input envelope")
        return self

    def to_instance(self) -> Instance:
        return Instance(
            targets=tuple(self.targets),
            min_delay=self.minDelay,
            max_delay=self.maxDelay,
            max_step=self.maxStep,
            max_ramps=self.maxRamps,
            anchors=tuple(Anchor(element=a.element, delay=a.delay)
                          for a in self.anchors),
        )
