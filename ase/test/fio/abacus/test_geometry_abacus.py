import numpy as np
import pytest

from ase.build import bulk
from ase.constraints import FixCartesian
from ase.io.abacus import read_abacus as read
from ase.units import Bohr

format = "abacus"

file = "STRU"


@pytest.fixture
def Si():
    return bulk("Si")


def test_cartesian_Si(Si):
    """write cartesian coords and check if structure was preserved"""
    Si.write(file, format=format, scaled=False)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_scaled_Si(Si):
    """write fractional coords and check if structure was preserved"""
    Si.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_constraints_Si(Si):
    """Test that non-parmetric constraints are written and read in properly"""
    Si.set_constraint([FixCartesian(0, [1, 0, 1]), FixCartesian(1, [1, 0, 1])])
    Si.write(file, format=format, scaled=True)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)
    assert len(Si.constraints) == len(new_atoms.constraints)
    assert str(Si.constraints[0]) == str(new_atoms.constraints[0])
    assert str(Si.constraints[1]) == str(new_atoms.constraints[1])


def test_pp_basis_Si(Si):
    """Test that pseudopotential and basis are written and read in properly"""
    pp = {"Si": "Si_ONCV_PBE-1.0.upf"}
    basis = {"Si": "dpsi_Si.dat"}
    Si.write(file, format=format, pp=pp, basis=basis)
    new_atoms, atom_potential, atom_basis, atom_offsite_basis = read(file, verbose=True)
    assert len(new_atoms.info["pp"]) == 1
    assert len(new_atoms.info["basis"]) == 1
    assert new_atoms.info["pp"]["Si"] == "Si_ONCV_PBE-1.0.upf"
    assert new_atoms.info["basis"]["Si"] == "dpsi_Si.dat"
    assert not atom_offsite_basis


stru_lines = """
ATOMIC_SPECIES
Si 1.000 Si.pz-vbc.UPF  #Element, Mass, Pseudopotential

NUMERICAL_ORBITAL
./Si_lda_8.0au_50Ry_2s2p1d

LATTICE_CONSTANT
10.2

ATOMIC_POSITIONS
Cartesian               #Cartesian(Unit is LATTICE_CONSTANT)
Si                      #Name of element
0.0                     #Magnetic for this element.
2                       #Number of atoms
0.00 0.00 0.00 0 0 0    #x,y,z, move_x, move_y, move_z
0.25 0.25 0.25 1 1 1
"""

expected_cell = np.array([[-0.5, 0, 0.5], [0, 0.5, 0.5], [-0.5, 0.5, 0]]) * 10.2 * Bohr


def test_latname_Si():
    latname = "fcc"
    with open(file, "w") as fd:
        fd.write(stru_lines)
    new_atoms = read(file, latname)
    assert np.allclose(new_atoms.get_cell(), expected_cell)
