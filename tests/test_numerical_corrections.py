"""Regression tests for the corrections to the synthetic study and tangent FPCA."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "methods")]
from engine import load
from srvf_numerics import tangent_fpca


class NumericalCorrectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(work / "matplotlib")
        cls.env, _ = load("synthetic", work / "output")
        cls.theta = 2 * np.pi * cls.env["s"]
        cls.contour = np.column_stack([np.cos(cls.theta), np.sin(cls.theta)])
        cls.profile = 0.2 + 0.05 * np.sin(cls.theta)
        cls.embedding = cls.env["embed_curve"](cls.contour, cls.profile)
        cls.q = cls.env["normalize_srvf"](cls.env["srvf"](cls.embedding))

    def test_curvature_includes_a_single_periodic_seam(self):
        estimate = self.env["compute_curvature_periodic"](self.contour)
        np.testing.assert_allclose(estimate, 1.0, atol=2e-4, rtol=0)
        self.assertEqual(estimate[-1], estimate[0])
        np.testing.assert_allclose(estimate[0], estimate[1], atol=1e-12, rtol=0)

    def test_smoothing_wraps_the_unique_nodes(self):
        values = np.array([4.0, 0.0, 0.0, 0.0, 4.0])
        result = self.env["circular_smooth"](values, n_iter=1)
        np.testing.assert_array_equal(result, [2, 1, 0, 1, 2])

    def test_inverse_restores_the_profile_scale_and_level(self):
        _, _, recovered = self.env["reconstruct_curves_from_srvfs"](
            self.q[None], self.embedding[None]
        )
        np.testing.assert_allclose(recovered[0], self.profile, atol=2e-8, rtol=0)

    def test_fpca_comparison_also_restores_the_profile_scale(self):
        contours, profiles = [], []
        for dx, dy in [(0.005, 0.004), (-0.005, 0.004), (0.005, -0.004), (-0.005, -0.004)]:
            contours.append(self.contour * [1 + dx, 1 + dy])
            curves = dx * np.cos(2 * self.theta) + dy * np.sin(3 * self.theta)
            profiles.append(self.profile + curves)
        embeddings = np.array([
            self.env["embed_curve"](b, r) for b, r in zip(contours, profiles)
        ])
        payload = {
            "mu": self.q,
            "pcs": np.zeros((1, self.q.size)),
            "scores": np.zeros((len(contours), 1)),
        }
        result = self.env["compare_aligned_fpca_vs_srvf_tangent_fpca"](
            np.array(contours), np.array(profiles), embeddings, payload, [1]
        )
        self.assertLess(result.iloc[0].thickness_error_srvf_tangent_fpca, 0.006)
        self.assertGreater(result.iloc[0].thickness_error_aligned_fpca, 0.0)

    def test_tangent_second_moment_uses_divisor_n(self):
        mu = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        values = np.array([[-2, 0], [-1, 0], [0, -1], [0, 1], [2, 0]])
        tangents = np.zeros((len(values), *mu.shape))
        tangents[:, :, 1:] = values[:, None, :]
        result = tangent_fpca(tangents, mu, lambda observation, _: observation)
        np.testing.assert_allclose(result["all_eigenvalues"][:2], [1.8, 0.4])
        self.assertAlmostEqual(result["all_eigenvalues"].sum(), 2.2)


if __name__ == "__main__":
    unittest.main()
