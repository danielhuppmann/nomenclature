import logging
import textwrap
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

import pandas as pd
import yaml
from pyam import IAMC_IDX, IamDataFrame
from pyam.logging import adjust_log_level
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nomenclature.definition import DataStructureDefinition
from nomenclature.error import ErrorCollector
from nomenclature.processor import Processor
from nomenclature.processor.utils import get_relative_path

logger = logging.getLogger(__name__)


class WarningEnum(IntEnum):
    error = 50
    high = 40
    medium = 30
    low = 20


class DataValidationCriteria(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    warning_level: WarningEnum = WarningEnum.error

    @field_validator("warning_level", mode="before")
    def validate_warning_level(cls, value):
        if isinstance(value, str):
            try:
                return WarningEnum[value]
            except KeyError:
                raise ValueError(
                    f"Invalid warning level: {value}. Expected one of:"
                    f" {', '.join(level.name for level in WarningEnum)}"
                )
        return value

    @property
    @abstractmethod
    def validation_args(self) -> dict[str, float | None]:
        pass

    def __str__(self):
        return ", ".join(
            [
                f"{key}: {value}"
                for key, value in self.model_dump(
                    exclude_none=True, exclude_unset=True, exclude={"warning_level"}
                ).items()
            ]
        )


class DataValidationValue(DataValidationCriteria):
    value: float
    rtol: float = 0.0
    atol: float = 0.0

    @property
    def tolerance(self) -> float:
        return self.value * self.rtol + self.atol

    @property
    def upper_bound(self) -> float:
        return self.value + self.tolerance

    @property
    def lower_bound(self) -> float:
        return self.value - self.tolerance

    @property
    def validation_args(self) -> dict[str, float | None]:
        return {"upper_bound": self.upper_bound, "lower_bound": self.lower_bound}


class DataValidationBounds(DataValidationCriteria):

    upper_bound: float | None = None
    lower_bound: float | None = None

    @model_validator(mode="after")
    def check_validation_criteria_exist(self):
        if self.upper_bound is None and self.lower_bound is None:
            raise ValueError("No validation criteria provided: " + str(self))
        return self

    @property
    def validation_args(self) -> dict[str, float | None]:
        return {"upper_bound": self.upper_bound, "lower_bound": self.lower_bound}


class DataValidationRange(DataValidationCriteria):
    range: list[float] = Field(..., min_length=2, max_length=2)

    @field_validator("range", mode="after")
    def check_range_is_valid(cls, value: list[float]):
        if value[0] > value[1]:
            raise ValueError(
                "Validation 'range' must be given as `(lower_bound, upper_bound)`, "
                "found: " + str(value)
            )
        return value

    @property
    def upper_bound(self) -> float:
        return self.range[1]

    @property
    def lower_bound(self) -> float:
        return self.range[0]

    @property
    def validation_args(self):
        """Attributes used for validation (as bounds)."""
        return {"upper_bound": self.upper_bound, "lower_bound": self.lower_bound}


class DataValidationItem(BaseModel):

    model: list[str] | None = None
    scenario: list[str] | None = None
    region: list[str] | None = None
    variable: list[str] | None = None
    unit: list[str] | None = None
    year: list[int] | None = None
    validation: list[DataValidationValue | DataValidationRange | DataValidationBounds]

    @field_validator("*", mode="before")
    @classmethod
    def single_input_to_list(cls, v):
        return v if isinstance(v, list) else [v]

    @field_validator("validation", mode="before")
    @classmethod
    def check_argument_mixing(cls, v, info):
        for item in v:
            pass
        return v

    @model_validator(mode="after")
    def check_warnings_order(self):
        """Check if warnings are set in descending order of severity."""
        if self.validation != sorted(
            self.validation, key=lambda c: c.warning_level, reverse=True
        ):
            raise ValueError(
                f"Validation criteria for {self.criteria} not sorted"
                " in descending order of severity."
            )
        else:
            return self

    @property
    def criteria(self):
        return self.model_dump(exclude_none=True, exclude_unset=True)

    @property
    def filter_args(self):
        """Attributes used for validation (as specified in the file)."""
        return self.model_dump(
            exclude_none=True, exclude_unset=True, exclude=["validation"]
        )

    def validate_with_definition(self, dsd: DataStructureDefinition) -> None:
        error_msg = ""

        # check for filter-items that are not defined in the codelists
        for dimension in IAMC_IDX:
            codelist = getattr(dsd, dimension, None)
            # no validation if codelist is not defined or filter-item is None
            if codelist is None or getattr(self, dimension) is None:
                continue
            if invalid := codelist.validate_items(getattr(self, dimension)):
                error_msg += (
                    f"The following {dimension}s are not defined in the "
                    "DataStructureDefinition:\n   " + ", ".join(invalid) + "\n"
                )

        if error_msg:
            raise ValueError(error_msg)

    def __str__(self):
        return ", ".join([f"{key}: {value}" for key, value in self.filter_args.items()])


class DataValidator(Processor):
    """Processor for validating IAMC datapoints"""

    criteria_items: list[DataValidationItem]
    file: Path

    @classmethod
    def from_file(cls, file: Path | str) -> "DataValidator":
        file = Path(file) if isinstance(file, str) else file
        with open(file, "r", encoding="utf-8") as f:
            criteria_items = yaml.safe_load(f)
        return cls(file=file, criteria_items=criteria_items)

    def apply(self, df: IamDataFrame) -> IamDataFrame:
        fail_list = []
        error = False

        with adjust_log_level():
            for item in self.criteria_items:
                per_item_df = df.filter(**item.filter_args)
                for criterion in item.validation:
                    failed_validation = per_item_df.validate(
                        **criterion.validation_args
                    )
                    if failed_validation is not None:
                        per_item_df = IamDataFrame(
                            pd.concat(
                                [per_item_df.data, failed_validation]
                            ).drop_duplicates(keep=False)
                        )
                        failed_validation["warning_level"] = (
                            criterion.warning_level.name
                        )
                        if criterion.warning_level == WarningEnum.error:
                            error = True
                        fail_list.append(
                            "  Criteria: " + str(item) + ", " + str(criterion)
                        )
                        fail_list.append(
                            textwrap.indent(failed_validation.to_string(), prefix="  ")
                            + "\n"
                        )

            fail_msg = "(file %s):\n" % get_relative_path(self.file)
            if error:
                fail_msg = (
                    "Data validation with error(s)/warning(s) "
                    + fail_msg
                    + "\n".join(fail_list)
                )
                logger.error(fail_msg)
                raise ValueError(
                    "Data validation failed. Please check the log for details."
                )
            if fail_list:
                fail_msg = (
                    "Data validation with warning(s) " + fail_msg + "\n".join(fail_list)
                )
                logger.warning(fail_msg)
        return df

    def validate_with_definition(self, dsd: DataStructureDefinition) -> None:
        errors = ErrorCollector(description=f"in file '{self.file}'")
        for criterion in self.criteria_items:
            try:
                criterion.validate_with_definition(dsd)
            except ValueError as value_error:
                errors.append(value_error)
        if errors:
            raise ValueError(errors)
