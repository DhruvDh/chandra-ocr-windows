"""Hardware-independent service contract. Health must never initialize a model."""
from typing import Protocol, Iterable
from threading import Event

class Backend(Protocol):
    def health(self) -> dict: ...
    def generate(self, request: dict, cancel: Event) -> Iterable[str | dict]: ...
