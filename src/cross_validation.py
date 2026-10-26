"""Deterministic flight-disjoint assignments and LOAFO entry point."""

from dataclasses import asdict, dataclass

from .config import AppConfig
from .errors import DataError


@dataclass(frozen=True)
class FlightSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = (self.train, self.validation, self.test)
        if any(not group or len(set(group)) != len(group) for group in groups):
            raise DataError("Each flight split must be nonempty and contain unique flight IDs.")
        if any(not isinstance(flight, str) or not flight.strip() for group in groups for flight in group):
            raise DataError("Flight split IDs must be nonempty strings.")
        if set(self.train) & set(self.validation) or set(self.train) & set(self.test) or set(self.validation) & set(self.test):
            raise DataError("Flight group overlap between train, validation, and reserved test.")

    def as_dict(self) -> dict:
        self.validate()
        return {key: list(value) for key, value in asdict(self).items()}


def pilot_split(eligible: list[str], minimum_flights: int = 3) -> FlightSplit:
    if len(set(eligible)) != len(eligible):
        raise DataError("Eligible flight IDs contain duplicates.")
    if len(eligible) < max(3, minimum_flights):
        raise DataError(f"Pilot needs at least {max(3, minimum_flights)} activity/reference-ready flights.")
    flights = sorted(eligible)
    split = FlightSplit(tuple(flights[2:]), (flights[1],), (flights[0],))
    split.validate()
    return split


def cross_validate(config: AppConfig) -> dict:
    from .loafo import run_loafo
    return run_loafo(config)


def loafo_splits(eligible: list[str], minimum_flights: int = 3) -> list[FlightSplit]:
    pilot_split(eligible, minimum_flights)  # shared distinct/minimum-group checks
    flights = sorted(eligible)
    result = []
    for index, test in enumerate(flights):
        validation = flights[(index + 1) % len(flights)]
        split = FlightSplit(tuple(f for f in flights if f not in (test, validation)), (validation,), (test,))
        split.validate()
        result.append(split)
    return result
