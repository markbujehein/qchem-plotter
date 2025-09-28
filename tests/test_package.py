from __future__ import annotations

import matplotlib as mpl

# Force a non-interactive backend for Matplotlib to prevent tkinter errors in CI
mpl.use("Agg")

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

# Import the classes from your library
import qchem_plotter as m
from qchem_plotter.orca_plotter import SpectrumPlotter, VibrationalSpectrum

# --- Existing Sanity Check ---------------------------------------------------


def test_version():
    """Tests that the installed version matches the package's reported version."""
    assert importlib.metadata.version("qchem_plotter") == m.__version__


# --- Test Fixtures -----------------------------------------------------------


@pytest.fixture
def sample_stk_file(tmp_path: Path) -> Path:
    """
    Creates a temporary, valid .stk file for testing.
    This fixture is automatically called by tests that use 'sample_stk_file'
    as an argument. `tmp_path` is a built-in pytest fixture that provides
    a temporary directory unique to the test function invocation.
    """
    stk_content = """
    100.0   10.0
    200.0   50.0
    300.0   25.0
    """
    file_path = tmp_path / "test.ir.stk"
    file_path.write_text(stk_content)
    return file_path


# --- Tests for VibrationalSpectrum Class -------------------------------------


def test_spectrum_loading_success(sample_stk_file: Path):
    """
    Tests successful loading and parsing of a valid .stk file.
    """
    spectrum = VibrationalSpectrum(sample_stk_file)

    assert len(spectrum) == 3
    assert spectrum.scaling_factor == 1.0
    np.testing.assert_array_equal(spectrum.frequencies, np.array([100.0, 200.0, 300.0]))
    np.testing.assert_array_equal(spectrum.intensities, np.array([10.0, 50.0, 25.0]))


def test_spectrum_frequency_scaling(sample_stk_file: Path):
    """
    Tests that the frequency scaling factor is applied correctly.
    """
    scaling_factor = 0.95
    spectrum = VibrationalSpectrum(sample_stk_file, scaling_factor=scaling_factor)

    expected_frequencies = np.array([100.0, 200.0, 300.0]) * scaling_factor
    np.testing.assert_allclose(spectrum.frequencies, expected_frequencies)


def test_spectrum_file_not_found():
    """
    Tests that a FileNotFoundError is raised for a non-existent file.
    """
    with pytest.raises(FileNotFoundError, match="Stick file not found"):
        VibrationalSpectrum("non_existent_file.stk")


# --- Tests for SpectrumPlotter Class -----------------------------------------


def test_plotter_initialization():
    """
    Tests that the SpectrumPlotter can be initialized without error.
    """
    plotter = SpectrumPlotter(x_min=0, x_max=4000, num_points=1000, fwhm=20)
    assert plotter.ir_spectrum is None
    assert plotter.raman_spectrum is None
    assert plotter.x_axis.shape == (1000,)


def test_plotter_add_spectrum(sample_stk_file: Path):
    """
    Tests that spectra can be added to the plotter correctly.
    """
    plotter = SpectrumPlotter(x_min=0, x_max=4000, num_points=1000, fwhm=20)
    ir_spec = VibrationalSpectrum(sample_stk_file)

    plotter.add_ir_spectrum(ir_spec)

    assert plotter.ir_spectrum is ir_spec
    assert plotter.raman_spectrum is None


def test_plotter_plot_method_runs(sample_stk_file: Path):
    """
    Tests that the main plot() method executes without raising an error
    and correctly configures the plot title.
    """
    plotter = SpectrumPlotter(x_min=0, x_max=4000, num_points=1000, fwhm=20)
    ir_spec = VibrationalSpectrum(sample_stk_file)
    plotter.add_ir_spectrum(ir_spec)

    plot_title = "Test Plot Title"

    # This block will fail if plotter.plot() raises any exceptions
    try:
        plotter.plot(title=plot_title)
    except Exception as e:
        pytest.fail(f"plotter.plot() raised an unexpected exception: {e}")

    # Indirectly test that the plotting logic ran by checking a side effect
    assert plotter.ax1.get_title() == plot_title
