from __future__ import annotations

import importlib.metadata

import qchem_plotter as m


def test_version():
    assert importlib.metadata.version("qchem_plotter") == m.__version__
