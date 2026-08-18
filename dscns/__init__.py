"""Dynamic Self-Modifying Cognitive Network System (DSCNS).

A faithful implementation of the DSCNS design report:
    experience -> multi-network observation -> independent evaluation ->
    cross-network verification -> selective internalization ->
    structural evolution -> continuous learning.

Phase 0 + Phase 1 prototype scope: a 100M-500M base language model with
5-6 cognitive networks, continual-learning experiments (Control / Exp1 / Exp2).
Phase 4: learned structural self-adaptation (policy-driven ArchitectureActions).
Phase 5: intrinsic parameter self-modification
        (theta -> h -> delta_theta -> theta', IntrinsicPlasticityModule).
"""

__version__ = "0.3.0"

from .intrinsic_plasticity import IntrinsicPlasticityModule
from .modification_memory import ModificationMemory
from .plasticity_trainer import PlasticityTrainer
from .self_modification import (ArchitectureAction, SelfModificationController,
                                SelfModificationPolicy, SelfStateEncoder)

__all__ = [
    "DSCNSSystem",
    "BaseLanguageModel",
    "CognitiveNetwork",
    "VerificationNetwork",
    "InternalizationController",
    "MetaCognitiveController",
    "MemorySystem",
    "NetworkCommunicationBus",
    "SelfModificationController",
    "SelfModificationPolicy",
    "SelfStateEncoder",
    "ArchitectureAction",
    "ModificationMemory",
    "IntrinsicPlasticityModule",
    "PlasticityTrainer",
]
