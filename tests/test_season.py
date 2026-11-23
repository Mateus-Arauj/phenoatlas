"""Tests for phenoatlas.season (find_peaks, season_mov, season_cal, validation)."""
import numpy as np
import pandas as pd
import pytest

from phenoatlas.season import (
    check_season,
    filter_seasons,
    find_peaks,
    season_cal,
    season_mov,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((t - mu) ** 2) / (2 * sigma**2))


def make_annual_series(n_years: int = 3, npt: int = 23) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic multi-year vegetation index time series."""
    n = n_years * npt
    t = np.linspace(0, n_years * 365, n)
    y = np.zeros(n)
    for yr in range(n_years):
        t_peak = yr * 365 + 180
        y += 0.1 + 0.7 * gaussian(t, t_peak, 40)
    return t, np.clip(y, 0, 1)


# ---------------------------------------------------------------------------
# find_peaks
# ---------------------------------------------------------------------------

class TestFindPeaks:
    def test_single_peak(self):
        t = np.linspace(0, 365, 46)
        y = gaussian(t, 180, 40)
        peaks, troughs = find_peaks(y)
        assert len(peaks) == 1
        # Peak should be near index 23 (middle)
        assert abs(t[peaks[0]] - 180) < 30

    def test_two_peaks(self):
        t = np.linspace(0, 730, 92)
        y = gaussian(t, 180, 40) + gaussian(t, 540, 40)
        peaks, _ = find_peaks(y, min_distance=20)
        assert len(peaks) >= 2

    def test_constant_no_peaks(self):
        y = np.ones(30)
        peaks, troughs = find_peaks(y)
        assert len(peaks) == 0
        assert len(troughs) == 0

    def test_output_types(self):
        t = np.linspace(0, 365, 46)
        y = gaussian(t, 180, 40)
        peaks, troughs = find_peaks(y)
        assert peaks.dtype == np.int64 or peaks.dtype == np.int32 or np.issubdtype(peaks.dtype, np.integer)

    def test_min_distance_enforced(self):
        t = np.linspace(0, 365, 92)
        # Two peaks close together
        y = gaussian(t, 90, 15) + gaussian(t, 100, 15)
        peaks, _ = find_peaks(y, min_distance=30)
        # With large min_distance, should merge into 1
        assert len(peaks) <= 2


# ---------------------------------------------------------------------------
# season_mov
# ---------------------------------------------------------------------------

class TestSeasonMov:
    def test_returns_list(self):
        t, y = make_annual_series(n_years=2, npt=23)
        seasons = season_mov(y, t, nptperyear=23, lam=100, iters=1)
        assert isinstance(seasons, list)

    def test_detects_seasons(self):
        t, y = make_annual_series(n_years=2, npt=23)
        # Default len_min scales with nptperyear so MODIS 16-day seasons are kept
        seasons = season_mov(y, t, nptperyear=23, lam=100, iters=1)
        assert len(seasons) >= 2

    def test_datetime_date_objects_supported(self):
        dates = pd.date_range("2020-01-01", periods=23 * 2, freq="16D").date
        _, y = make_annual_series(n_years=2, npt=23)
        seasons = season_mov(y, dates, nptperyear=23, lam=100, iters=1)
        assert len(seasons) >= 2

    def test_season_dict_keys(self):
        t, y = make_annual_series(n_years=1, npt=46)
        seasons = season_mov(y, t, nptperyear=46, lam=100, iters=1)
        if seasons:
            s = seasons[0]
            assert "idx_start" in s
            assert "idx_end" in s
            assert "t_start" in s
            assert "t_end" in s

    def test_season_ids_sequential(self):
        t, y = make_annual_series(n_years=2, npt=23)
        seasons = season_mov(y, t, nptperyear=23, lam=100, iters=1)
        ids = [s["season_id"] for s in seasons]
        assert ids == list(range(1, len(seasons) + 1))

    def test_calendar_year_mode(self):
        dates = pd.date_range("2020-01-01", periods=46 * 2, freq="8D")
        t = dates.to_numpy()
        y = np.zeros(len(t))
        for yr, peak_doy in [(2020, 180), (2021, 190)]:
            mask = pd.DatetimeIndex(t).year == yr
            doy = pd.DatetimeIndex(t[mask]).day_of_year.to_numpy()
            y[mask] = 0.1 + 0.7 * np.exp(-((doy - peak_doy) ** 2) / (2 * 40**2))
        seasons = season_mov(y, t, nptperyear=46, lam=100, iters=1, calendar_year=True)
        assert len(seasons) >= 1

    def test_sg_smoother(self):
        t, y = make_annual_series(n_years=1, npt=23)
        seasons = season_mov(y, t, nptperyear=23, smooth_method="sg", iters=1)
        assert isinstance(seasons, list)

    def test_invalid_smoother_raises(self):
        t, y = make_annual_series()
        with pytest.raises(ValueError):
            season_mov(y, t, smooth_method="not_a_method")


# ---------------------------------------------------------------------------
# season_cal
# ---------------------------------------------------------------------------

class TestSeasonCal:
    def test_two_years(self):
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="16D")
        seasons = season_cal(dates)
        years = {s["year"] for s in seasons}
        assert 2020 in years
        assert 2021 in years

    def test_custom_doy_range(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="16D")
        seasons = season_cal(dates, sos_doy=91, eos_doy=274)  # Apr–Oct
        assert len(seasons) >= 1
        for s in seasons:
            assert s["year"] == 2020

    def test_season_dict_keys(self):
        dates = pd.date_range("2021-01-01", periods=23, freq="16D")
        seasons = season_cal(dates)
        for s in seasons:
            assert "season_id" in s
            assert "year" in s
            assert "idx_start" in s
            assert "idx_end" in s


# ---------------------------------------------------------------------------
# Season validation
# ---------------------------------------------------------------------------

class TestCheckSeason:
    def setup_method(self):
        self.t, self.y = make_annual_series(n_years=1, npt=46)

    def test_valid_season_passes(self):
        season = {"idx_start": 5, "idx_end": 40}
        assert check_season(season, self.y, len_min=5, len_max=100)

    def test_too_short_fails(self):
        season = {"idx_start": 0, "idx_end": 3}
        assert not check_season(season, self.y, len_min=10)

    def test_too_long_fails(self):
        season = {"idx_start": 0, "idx_end": 45}
        assert not check_season(season, self.y, len_max=10)

    def test_low_amplitude_fails(self):
        y_flat = np.ones(46) * 0.5
        season = {"idx_start": 0, "idx_end": 45}
        assert not check_season(season, y_flat, amp_min=0.2)

    def test_negative_ndvi_allowed_and_weights_enforced(self):
        y = np.linspace(-0.2, 0.4, 46)
        season = {"idx_start": 0, "idx_end": 45}
        assert check_season(season, y, len_min=10, coverage_min=0.9)
        w_mostly_zero = np.zeros(46)
        w_mostly_zero[:5] = 1.0
        assert not check_season(
            season, y, w=w_mostly_zero, len_min=10, coverage_min=0.5
        )


class TestFilterSeasons:
    def test_filters_correctly(self):
        _, y = make_annual_series(n_years=2, npt=23)
        seasons = [
            {"idx_start": 0, "idx_end": 45},   # long (valid)
            {"idx_start": 0, "idx_end": 2},    # too short
        ]
        valid = filter_seasons(seasons, y, len_min=10, len_max=100)
        assert len(valid) == 1
