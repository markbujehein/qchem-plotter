#!/usr/bin/env python3
"""
A Python tool for plotting publication-quality vibrational spectra from ORCA .stk files.

This script reads frequency and intensity data from ORCA's stick spectrum files
(\\*.ir.stk, \\*.raman.stk), applies Gaussian broadening to simulate a lineshape,
and plots the resulting IR (transmittance) and Raman (intensity) spectra on a
shared axis.

The tool is designed to be run from the command line, providing a flexible
interface for customization of plotting parameters.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class VibrationalSpectrum:
    """
    Represents a single vibrational spectrum from an ORCA .stk file.

    This class loads the stick spectrum (frequencies and intensities),
    handles frequency scaling, and can generate a convoluted (broadened)
    spectrum using a Gaussian lineshape.

    Attributes:
        filepath (Path): The path to the input .stk file.
        scaling_factor (float): The frequency scaling factor to apply.
        raw_sticks (np.ndarray): The raw data loaded from the file (N, 2).
        frequencies (np.ndarray): The scaled vibrational frequencies in cm-1.
        intensities (np.ndarray): The corresponding intensities/activities.
    """

    def __init__(self, stk_file: str | Path, scaling_factor: float = 1.0):
        """
        Initializes the VibrationalSpectrum instance.

        Args:
            stk_file: Path to the .stk file.
            scaling_factor: A multiplicative factor to scale frequencies.

        Raises:
            FileNotFoundError: If the specified stk_file does not exist.
        """
        self.filepath = Path(stk_file)
        if not self.filepath.is_file():
            msg = f"Stick file not found: {self.filepath}"
            raise FileNotFoundError(msg)

        self.scaling_factor = scaling_factor
        self.raw_sticks: np.ndarray = self._load_stk()
        self.frequencies: np.ndarray = self.raw_sticks[:, 0] * self.scaling_factor
        self.intensities: np.ndarray = self.raw_sticks[:, 1]

    def _load_stk(self) -> np.ndarray:
        """Loads the stick data from the file."""
        try:
            # Cast the result to tell mypy we are confident it's an ndarray
            return cast(np.ndarray, np.loadtxt(self.filepath))
        except OSError as e:
            logging.error("Error: Could not read the file %s", self.filepath)
            raise e

    def __len__(self) -> int:
        """Returns the number of vibrational modes."""
        return len(self.frequencies)

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the object."""
        return (
            f"{self.__class__.__name__}("
            f"file='{self.filepath.name}', "
            f"modes={len(self)}, "
            f"scaled_by={self.scaling_factor}"
            ")"
        )

    def convolve(self, x_axis: np.ndarray, fwhm: float) -> np.ndarray:
        """
        Generates a convoluted spectrum using Gaussian broadening.

        Args:
            x_axis: The wavenumber axis (numpy array) for the convoluted spectrum.
            fwhm: The full-width at half-maximum for the Gaussian peaks in cm-1.

        Returns:
            A numpy array representing the convoluted y-axis (intensity).
        """
        y_axis = np.zeros_like(x_axis)
        # The standard deviation (sigma) of a Gaussian is related to FWHM by:
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        for freq, intensity in zip(self.frequencies, self.intensities):
            # Sum of Gaussian functions centered at each vibrational frequency
            y_axis += intensity * np.exp(-((x_axis - freq) ** 2) / (2 * sigma**2))
        # Cast the result to assure mypy of the final type
        return cast(np.ndarray, y_axis)


class SpectrumPlotter:
    """
    Handles the creation and configuration of the combined spectra plot.

    This class sets up the Matplotlib figure and axes, plots the provided
    IR and Raman spectra with appropriate transformations (e.g., transmittance),
    and configures all plot elements like labels, titles, and legends.
    """

    # Conversion factor from cm-1 to eV. 1 eV = 8065.54 cm-1
    CM_TO_EV = 1 / 8065.54

    def __init__(self, x_min: float, x_max: float, num_points: int, fwhm: float):
        """
        Initializes the SpectrumPlotter.

        Args:
            x_min: Minimum value for the x-axis (wavenumber).
            x_max: Maximum value for the x-axis (wavenumber).
            num_points: Number of points to generate for the x-axis.
            fwhm: FWHM for peak broadening, passed to spectrum convolution.
        """
        self.x_axis = np.linspace(x_min, x_max, num_points)
        self.fwhm = fwhm
        self.ir_spectrum: VibrationalSpectrum | None = None
        self.raman_spectrum: VibrationalSpectrum | None = None

        self.fig, self.ax1 = plt.subplots(figsize=(16, 8))
        self.ax2 = self.ax1.twinx()

    def add_ir_spectrum(self, spectrum: VibrationalSpectrum) -> None:
        """Adds an IR spectrum to be plotted."""
        self.ir_spectrum = spectrum

    def add_raman_spectrum(self, spectrum: VibrationalSpectrum) -> None:
        """Adds a Raman spectrum to be plotted."""
        self.raman_spectrum = spectrum

    def _plot_ir(self) -> None:
        """Internal method to process and plot the IR spectrum."""
        if not self.ir_spectrum:
            return

        y_raw = self.ir_spectrum.convolve(self.x_axis, self.fwhm)
        y_max = np.max(y_raw) if np.max(y_raw) > 1e-9 else 1.0

        # Convert IR from absorption to transmittance [0, 1]
        y_transmittance = 1.0 - (y_raw / y_max)
        # Apply a vertical offset for display purposes [1.2, 2.2]
        y_offset = y_transmittance + 1.2

        self.ax1.plot(
            self.x_axis, y_offset, color="red", lw=1.5, label="IR (Transmittance)"
        )
        self.ax1.set_ylabel("Transmittance (a.u.)", color="red", fontsize=14)
        self.ax1.tick_params(axis="y", labelcolor="red", labelsize=12)

        # Manually set y-ticks to represent the original 0-1 range
        self.ax1.set_yticks([1.2, 1.7, 2.2])
        self.ax1.set_yticklabels(["0.0", "0.5", "1.0"])

    def _plot_raman(self) -> None:
        """Internal method to process and plot the Raman spectrum."""
        if not self.raman_spectrum:
            return

        y_raw = self.raman_spectrum.convolve(self.x_axis, self.fwhm)
        y_max = np.max(y_raw) if np.max(y_raw) > 1e-9 else 1.0

        # Normalize Raman intensity to the range [0, 1]
        y_normalized = y_raw / y_max

        self.ax2.plot(
            self.x_axis, y_normalized, color="blue", lw=1.5, label="Raman (Intensity)"
        )
        self.ax2.set_ylabel("Raman Intensity (Normalized)", color="blue", fontsize=14)
        self.ax2.tick_params(axis="y", labelcolor="blue", labelsize=12)
        self.ax2.set_yticks([0.0, 0.5, 1.0])

    def configure_plot(self, title: str) -> None:
        """
        Configures the final plot elements (axes, labels, titles, legend).

        Args:
            title: The main title for the plot.
        """
        y_min, y_max = -0.1, 2.3
        self.ax1.set_ylim(y_min, y_max)
        self.ax2.set_ylim(y_min, y_max)

        self.ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
        self.ax1.set_xlim(
            self.x_axis[-1], self.x_axis[0]
        )  # Invert for conventional view
        self.ax1.tick_params(axis="x", labelsize=12)

        # Configure the top x-axis for Energy in eV
        ax_top = self.ax1.secondary_xaxis(
            "top", functions=(lambda x: x * self.CM_TO_EV, lambda x: x / self.CM_TO_EV)
        )
        ax_top.set_xlabel("Energy (eV)", fontsize=14)
        ax_top.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax_top.tick_params(axis="x", labelsize=12)

        self.ax1.set_title(title, fontsize=16, weight="bold")

        # Combine legends from both y-axes into a single box
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(
            lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12
        )

        self.fig.tight_layout()

    def plot(self, title: str = "Calculated Vibrational Spectra") -> None:
        """
        Generates the full plot by calling the individual plotting methods.

        Args:
            title: The main title for the plot.

        Raises:
            ValueError: If no spectra have been added to the plotter.
        """
        if self.ir_spectrum is None and self.raman_spectrum is None:
            msg = "No spectra have been added to the plotter."
            raise ValueError(msg)

        if self.ir_spectrum:
            self._plot_ir()
        if self.raman_spectrum:
            self._plot_raman()

        self.configure_plot(title)

    def save(self, filename: str | Path, **kwargs: Any) -> None:
        """
        Saves the figure to a file.

        Args:
            filename: The path to save the output file.
            **kwargs: Additional keyword arguments passed to `fig.savefig()`.
        """
        self.fig.savefig(filename, **kwargs)
        logging.info("-> Plot saved to %s", filename)

    def show(self) -> None:
        """Displays the plot in an interactive window."""
        plt.show()


def main() -> None:
    """
    Main function to parse command-line arguments and run the plotter.
    """
    parser = argparse.ArgumentParser(
        description="Plot convoluted IR and/or Raman spectra from ORCA .stk files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ir", type=str, help="Path to the IR stick file (*.ir.stk).")
    parser.add_argument(
        "--raman", type=str, help="Path to the Raman stick file (*.raman.stk)."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="spectrum.pdf",
        help="Output file name for the plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Calculated Vibrational Spectra",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--fwhm", type=float, default=20.0, help="Full-width at half-maximum (cm-1)."
    )
    parser.add_argument(
        "--scale", type=float, default=0.975, help="Frequency scaling factor."
    )
    parser.add_argument(
        "--xmin", type=float, default=0.0, help="Minimum wavenumber (cm-1)."
    )
    parser.add_argument(
        "--xmax", type=float, default=4000.0, help="Maximum wavenumber (cm-1)."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=4.0,
        help="Points per cm-1 for convolution.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the plot interactively."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.ir and not args.raman:
        parser.error("At least one spectrum file (--ir or --raman) must be provided.")

    num_points = int((args.xmax - args.xmin) * args.resolution)

    plotter = SpectrumPlotter(
        x_min=args.xmin, x_max=args.xmax, num_points=num_points, fwhm=args.fwhm
    )

    try:
        if args.ir:
            ir_spec = VibrationalSpectrum(args.ir, scaling_factor=args.scale)
            plotter.add_ir_spectrum(ir_spec)
            logging.info("-> Loaded IR spectrum: %s", ir_spec)

        if args.raman:
            raman_spec = VibrationalSpectrum(args.raman, scaling_factor=args.scale)
            plotter.add_raman_spectrum(raman_spec)
            logging.info("-> Loaded Raman spectrum: %s", raman_spec)

    except (FileNotFoundError, OSError) as e:
        logging.critical("Fatal Error: %s", e)
        return  # Exit if a file cannot be loaded/read

    plotter.plot(title=args.title)
    plotter.save(args.output, bbox_inches="tight", transparent=True, dpi=300)

    if not args.no_show:
        plotter.show()


if __name__ == "__main__":
    main()
