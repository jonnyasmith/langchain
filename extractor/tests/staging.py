"""A provider stand-in for tests that drive `main` rather than an adapter.

`main` depends on the `Provider` protocol, not on the registry's `ProviderAdapter`, so a test
can satisfy that protocol without constructing a chat model or naming a provider SDK. Shared by
`test_cli.py` and `test_live.py`, which both need it and would otherwise copy it.
"""

from collections.abc import Callable

from extractor.extraction import ExtractionPort, PortSettings

type PortFactory = Callable[[PortSettings], ExtractionPort]
"""Construct a port from settings. Only tests name this now: production reaches a port through
`Provider.build_port`, which is the seam `main` depends on."""


class StagedProvider:
    """Satisfies the `Provider` protocol from a default model and a port factory."""

    def __init__(self, default_model: str, build_port: PortFactory) -> None:
        self.default_model = default_model
        self._build_port = build_port

    def build_port(self, settings: PortSettings) -> ExtractionPort:
        return self._build_port(settings)
