"""Minimal state-based BC actor + sibling forward model.

The chunked/squashed/visual flavor lives in src/legacy/actor/.
"""

from src.actor.forward import ForwardModel
from src.actor.model import Actor

__all__ = ["Actor", "ForwardModel"]
