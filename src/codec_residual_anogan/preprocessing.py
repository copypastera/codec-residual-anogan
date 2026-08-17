"""Source-only scaling and PCA preprocessing."""
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from .utils import atomic_json


class FeaturePreprocessor(object):
    """Standardize optional PCA inputs using only normal training rows."""

    VALID = {
        "none", "standard", "robust", "l2", "standard_l2", "robust_l2"}

    def __init__(self, kind="standard", pca=None, eps=1e-8):
        if kind not in self.VALID:
            raise ValueError("unknown preprocessing kind %r" % kind)
        self.kind = kind
        self.pca_spec = pca
        self.eps = float(eps)
        self.center = None
        self.scale = None
        self.pca = None
        self.n_fit = 0
        self.fit_id_sha256 = None

    @property
    def uses_l2(self):
        return self.kind in ("l2", "standard_l2", "robust_l2")

    @property
    def scaling_kind(self):
        if self.kind in ("standard", "standard_l2"):
            return "standard"
        if self.kind in ("robust", "robust_l2"):
            return "robust"
        return "none"

    def fit(self, features, fit_ids):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("preprocessor expects an N x D matrix")
        if self.scaling_kind == "standard":
            self.center = values.mean(axis=0)
            self.scale = values.std(axis=0)
        elif self.scaling_kind == "robust":
            self.center = np.median(values, axis=0)
            q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
            self.scale = q75 - q25
        else:
            self.center = np.zeros(values.shape[1], dtype=np.float64)
            self.scale = np.ones(values.shape[1], dtype=np.float64)
        self.scale = np.maximum(self.scale, self.eps)
        transformed = self._scale_and_l2(values)
        if self.pca_spec not in (None, "none", 0, 0.0):
            requested = self.pca_spec
            if isinstance(requested, str):
                requested = (
                    float(requested) if "." in requested else int(requested))
            if isinstance(requested, float) and not 0.0 < requested < 1.0:
                requested = int(requested)
            self.pca = PCA(
                n_components=requested, whiten=False, svd_solver="full")
            self.pca.fit(transformed)
        self.n_fit = int(len(values))
        self.fit_id_sha256 = hashlib.sha256(
            ("\n".join(fit_ids) + "\n").encode("utf-8")).hexdigest()
        return self

    def _scale_and_l2(self, features):
        values = (features - self.center) / self.scale
        if self.uses_l2:
            norms = np.sqrt(np.square(values).sum(axis=1, keepdims=True))
            values = values / np.maximum(norms, self.eps)
        return values

    def transform(self, features):
        if self.center is None:
            raise RuntimeError("preprocessor is not fitted")
        values = self._scale_and_l2(
            np.asarray(features, dtype=np.float64))
        if self.pca is not None:
            values = self.pca.transform(values)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("preprocessing produced non-finite values")
        return values

    @property
    def output_dim(self):
        if self.pca is None:
            return int(len(self.center))
        return int(self.pca.n_components_)

    def metadata(self):
        return {
            "kind": self.kind,
            "scaling_kind": self.scaling_kind,
            "l2_normalization": self.uses_l2,
            "pca_requested": self.pca_spec,
            "pca_output_dim": self.output_dim,
            "fit_rows": self.n_fit,
            "fit_id_sha256": self.fit_id_sha256,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "preprocessor.joblib"
        joblib.dump(self, artifact)
        atomic_json(directory / "preprocessor.json", self.metadata())
        return artifact
