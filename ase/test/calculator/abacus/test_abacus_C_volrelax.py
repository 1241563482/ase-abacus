import pytest
import numpy as np
from glob import glob
from ase import io
from ase.optimize import BFGS
from ase.build import bulk
from ase.calculators.abacus import get_abacus_version

calc = pytest.mark.calculator

version_string = """\

                             WELCOME TO ABACUS

               'Atomic-orbital Based Ab-initio Computation at UStc'

                     Website: http://abacus.ustc.edu.cn/

    Version: Parallel, in development
    Processor Number is 8
    Start Time is Sat Jun  4 18:01:24 2022
"""


def test_get_abacus_version():
    assert get_abacus_version(version_string) == 'Parallel, in development'


@calc('abacus')
def test_abacus_C_volrelax(factory):
    """
    Run ABACUS tests to ensure that relaxation with the ABACUS calculator works.

    """

    kpts = [4, 4, 4]
    input_param = {'ntype': 1, 'ecutwfc': 20, 'scf_nmax': 50, 'smearing_method': 'gaussian',
                   'smearing_sigma': 0.01, 'basis_type': 'lcao', 'ks_solver': 'genelpa',
                   'mixing_type': 'pulay', 'mixing_beta': 0.7, 'scf_thr': 1e-6, 'out_chg': 1,
                   'relax_method': 'bfgs', 'xc': 'pbe'}

    # -- Perform Volume relaxation within Vasp
    def abacus_vol_relax():
        C = bulk('C')
        calc = factory.calc(calculation='cell-relax', force_thr_ev=0.1,
                            stress_thr=10, cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        C.calc = calc
        C.get_potential_energy()  # Execute

        # Explicitly parse atomic position output file from Abacus
        out_files = glob('OUT.ABACUS/STRU_ION*_D')
        STRU_C = io.read(out_files[-1], format='abacus')

        r_C = io.read('OUT.ABACUS/running_cell-relax.log',
                      format='abacus-out')

        assert cells_almost_equal(r_C.get_cell(), STRU_C.get_cell())

        return r_C

    # -- Volume relaxation using ASE with Abacus as force/stress calculator
    def ase_vol_relax():
        C = bulk('C')
        calc = factory.calc(calculation='scf', cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        C.calc = calc

        from ase.constraints import StrainFilter
        sf = StrainFilter(C)
        with BFGS(sf, logfile='relaxation.log') as qn:
            qn.run(fmax=0.1, steps=5)

        r_C = io.read('OUT.ABACUS/running_scf.log',
                      format='abacus-out')

        return r_C

    # Test function for comparing two cells
    def cells_almost_equal(cellA, cellB, tol=0.01):
        return (np.abs(cellA - cellB) < tol).all()

    C_abacus = abacus_vol_relax()
    C_ase = ase_vol_relax()

    assert cells_almost_equal(C_ase.get_cell(), C_abacus.get_cell())
