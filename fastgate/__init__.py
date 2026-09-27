"""FastGate: a System One (Jev) decision layer for multilingual RAG helpdesks."""
from .core import Decision, decide, route
from .kb import KnowledgeBase

__all__ = ["Decision", "decide", "route", "KnowledgeBase"]
__version__ = "0.1.0"
