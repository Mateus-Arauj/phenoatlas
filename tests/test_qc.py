"""Tests for phenoatlas.qc (MODIS, Sentinel-2, NDVI3g, SPOT QC decoders)."""
import numpy as np
import pytest

from phenoatlas.qc import (
    qc_modis_lai,
    qc_modis_vi,
    qc_modis_vi_detailed,
    qc_ndvi3g,
    qc_sentinel2,
    qc_spot,
)


# ---------------------------------------------------------------------------
# qc_modis_vi (SummaryQA)
# ---------------------------------------------------------------------------

class TestQcModisVi:
    def test_good_data_weight_1(self):
        qc = np.array([0], dtype=np.uint16)
        w = qc_modis_vi(qc, level=1)
        assert w[0] == pytest.approx(1.0)

    def test_cloud_weight_0(self):
        qc = np.array([3], dtype=np.uint16)  # bits 00=cloud
        w = qc_modis_vi(qc, level=3)
        assert w[0] == pytest.approx(0.0)

    def test_marginal_level1_zero(self):
        qc = np.array([1], dtype=np.uint16)  # marginal
        w = qc_modis_vi(qc, level=1)
        assert w[0] == pytest.approx(0.0)

    def test_marginal_level2_half(self):
        qc = np.array([1], dtype=np.uint16)
        w = qc_modis_vi(qc, level=2)
        assert w[0] == pytest.approx(0.5)

    def test_output_shape(self):
        qc = np.arange(100, dtype=np.uint16)
        w = qc_modis_vi(qc)
        assert w.shape == (100,)

    def test_weights_in_range(self):
        qc = np.arange(4, dtype=np.uint16)
        w = qc_modis_vi(qc)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ---------------------------------------------------------------------------
# qc_modis_vi_detailed (DetailedQA)
# ---------------------------------------------------------------------------

class TestQcModisViDetailed:
    def test_output_shape(self):
        qc = np.zeros(50, dtype=np.uint16)
        w = qc_modis_vi_detailed(qc)
        assert w.shape == (50,)

    def test_best_quality_high_weight(self):
        qc = np.array([0], dtype=np.uint16)  # all zeros = best quality
        w = qc_modis_vi_detailed(qc)
        assert w[0] >= 0.8

    def test_weights_in_range(self):
        rng = np.random.default_rng(0)
        qc = rng.integers(0, 65535, 200, dtype=np.uint16)
        w = qc_modis_vi_detailed(qc)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ---------------------------------------------------------------------------
# qc_modis_lai (FparLai_QC)
# ---------------------------------------------------------------------------

class TestQcModisLai:
    def test_best_quality(self):
        qc = np.array([0b00000000], dtype=np.uint8)  # all zeros = good
        w = qc_modis_lai(qc)
        assert w[0] == pytest.approx(1.0)

    def test_snow_zero(self):
        qc = np.array([0b10000000], dtype=np.uint8)  # bit 7 = snow
        w = qc_modis_lai(qc)
        assert w[0] == pytest.approx(0.0)

    def test_not_produced_zero(self):
        qc = np.array([0b00100000], dtype=np.uint8)  # bit 5 = not produced
        w = qc_modis_lai(qc)
        assert w[0] == pytest.approx(0.0)

    def test_output_shape(self):
        qc = np.zeros(30, dtype=np.uint8)
        w = qc_modis_lai(qc)
        assert w.shape == (30,)

    def test_weights_in_range(self):
        qc = np.arange(256, dtype=np.uint8)
        w = qc_modis_lai(qc)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ---------------------------------------------------------------------------
# qc_sentinel2 (SCL)
# ---------------------------------------------------------------------------

class TestQcSentinel2:
    def test_vegetation_weight_1(self):
        scl = np.array([4], dtype=np.uint8)  # vegetation
        w = qc_sentinel2(scl)
        assert w[0] == pytest.approx(1.0)

    def test_cloud_weight_0(self):
        for cls in [8, 9, 10]:
            scl = np.array([cls], dtype=np.uint8)
            w = qc_sentinel2(scl)
            assert w[0] == pytest.approx(0.0), f"Cloud class {cls} should have weight 0"

    def test_shadow_weight_0(self):
        scl = np.array([3], dtype=np.uint8)
        w = qc_sentinel2(scl)
        assert w[0] == pytest.approx(0.0)

    def test_snow_weight_0(self):
        scl = np.array([11], dtype=np.uint8)
        w = qc_sentinel2(scl)
        assert w[0] == pytest.approx(0.0)

    def test_output_shape(self):
        scl = np.arange(12, dtype=np.uint8)
        w = qc_sentinel2(scl)
        assert w.shape == (12,)

    def test_weights_in_range(self):
        scl = np.arange(12, dtype=np.uint8)
        w = qc_sentinel2(scl)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ---------------------------------------------------------------------------
# qc_ndvi3g
# ---------------------------------------------------------------------------

class TestQcNdvi3g:
    def test_good_data_weight_1(self):
        qc = np.array([1], dtype=np.int8)
        w = qc_ndvi3g(qc)
        assert w[0] == pytest.approx(1.0)

    def test_missing_weight_0(self):
        for flag in [4, 5, 6]:
            qc = np.array([flag], dtype=np.int8)
            w = qc_ndvi3g(qc)
            assert w[0] == pytest.approx(0.0), f"Flag {flag} should → 0"

    def test_marginal_weight_05(self):
        qc = np.array([3], dtype=np.int8)  # cloudy
        w = qc_ndvi3g(qc)
        assert w[0] == pytest.approx(0.5)

    def test_output_shape(self):
        qc = np.array([1, 2, 3, 4, 5, 6], dtype=np.int8)
        w = qc_ndvi3g(qc)
        assert w.shape == (6,)


# ---------------------------------------------------------------------------
# qc_spot
# ---------------------------------------------------------------------------

class TestQcSpot:
    def test_good_weight_1(self):
        qc = np.array([0], dtype=np.int8)
        w = qc_spot(qc)
        assert w[0] == pytest.approx(1.0)

    def test_cloud_weight_0(self):
        for flag in [2, 3, 4, 5]:
            qc = np.array([flag], dtype=np.int8)
            w = qc_spot(qc)
            assert w[0] == pytest.approx(0.0)

    def test_output_shape(self):
        qc = np.arange(6, dtype=np.int8)
        w = qc_spot(qc)
        assert w.shape == (6,)

    def test_weights_in_range(self):
        qc = np.arange(6, dtype=np.int8)
        w = qc_spot(qc)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)
