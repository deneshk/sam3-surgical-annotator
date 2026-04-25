"""Research-mode helpers for experiment tracking."""

__all__ = []

try:
    from annotator.research.controller import ResearchController
except ModuleNotFoundError:
    ResearchController = None
else:
    __all__.append("ResearchController")
