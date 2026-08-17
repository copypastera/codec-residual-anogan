"""Compatibility alias for the released preprocessor pickle.

New artifacts are serialized from the package module. The released artifact
was serialized before the src-layout refactor and imports this module name.
"""

from codec_residual_anogan.preprocessing import FeaturePreprocessor

__all__ = ["FeaturePreprocessor"]
