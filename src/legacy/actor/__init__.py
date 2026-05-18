"""Legacy actor used by the parked dynamics-rollout path. Not part of the BC baseline."""

from src.legacy.actor.config import ActorConfig
from src.legacy.actor.model import Actor
from src.legacy.actor.training import VideoActorTrainer, VideoActorTrainerConfig

__all__ = [
    "Actor",
    "ActorConfig",
    "VideoActorTrainer",
    "VideoActorTrainerConfig",
]
