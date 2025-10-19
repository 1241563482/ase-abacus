import numpy as np
import pytest

from ase.io.abacus import AbacusOutCalcChunk, AbacusOutChunk, AbacusOutHeaderChunk
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import GPa

eps_hp = 1e-10  # The espsilon value used to compare numbers that are high-precision
eps_lp = 1e-7  # The espsilon value used to compare numbers that are low-precision


@pytest.fixture
def default_chunk():
    contents = """ READING UNITCELL INFORMATION
                                    ntype = 1
                 atom label for species 1 = Fe
                  lattice constant (Bohr) = 1.81387
              lattice constant (Angstrom) = 0.959861

 READING ATOM TYPE 1
                               atom label = Fe
                      L=0, number of zeta = 1
                      L=1, number of zeta = 1
                      L=2, number of zeta = 1
             number of atom for this type = 2
                      start magnetization = 5
                      start magnetization = 5

                        TOTAL ATOM NUMBER = 2

 CARTESIAN COORDINATES ( UNIT = 1.81387 Bohr ).
         atom                   x                   y                   z                 mag                  vx                  vy                  vz
     tauc_Fe1                   0                   0                   0                   5                   0                   0                   0
     tauc_Fe2            1.416755            1.416755            1.416755                   5                   0                   0                   0"""
    return AbacusOutChunk(contents)


def test_search_parse_scalar(default_chunk):
    assert default_chunk.parse_scalar(r"TOTAL ATOM NUMBER = (\d+)") == 2
    assert int(default_chunk.parse_scalar(r"ntype = (\d+)")) == 1


def test_coordinate_system(default_chunk):
    assert default_chunk.coordinate_system == "CARTESIAN"


@pytest.fixture
def initial_cell():
    return np.array(
        [
            [4.12959, -0.005768, -0.002402],
            [0.00403, 3.87003, -0.001863],
            [-0.000266, -0.001379, 4.14035],
        ]
    )


@pytest.fixture
def initial_positions():
    return np.array(
        [
            [0.02669, 0.982777, 3e-05],
            [0.023514, 0.483068, 0.500002],
            [0.522649, 0.981899, 0.499974],
            [0.527148, 0.482256, 0.999994],
        ]
    )


@pytest.fixture
def initial_velocities():
    return np.zeros((4, 3))


@pytest.fixture
def header_chunk():
    contents = """
READING GENERAL INFORMATION
                           global_out_dir = OUT.ABACUS/
                           global_in_card = INPUT
                               pseudo_dir =
                              orbital_dir =
                              pseudo_type = auto
                                    DRANK = 1
                                    DSIZE = 8
                                   DCOLOR = 1
                                    GRANK = 1
                                    GSIZE = 1




 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Reading atom information in unitcell:                              |
 | From the input file and the structure file we know the number of   |
 | different elments in this unitcell, then we list the detail        |
 | information for each element, especially the zeta and polar atomic |
 | orbital number for each element. The total atom number is counted. |
 | We calculate the nearest atom distance for each atom and show the  |
 | Cartesian and Direct coordinates for each atom. We list the file   |
 | address for atomic orbitals. The volume and the lattice vectors    |
 | in real and reciprocal space is also shown.                        |
 |                                                                    |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




 READING UNITCELL INFORMATION
                                    ntype = 1
                 atom label for species 1 = Al
                  lattice constant (Bohr) = 1.88973
              lattice constant (Angstrom) = 1

 READING ATOM TYPE 1
                               atom label = Al
                      L=0, number of zeta = 3
                      L=1, number of zeta = 3
                      L=2, number of zeta = 2
             number of atom for this type = 4
                      start magnetization = FALSE
                      start magnetization = FALSE
                      start magnetization = FALSE
                      start magnetization = FALSE

                        TOTAL ATOM NUMBER = 4

 DIRECT COORDINATES
         atom                   x                   y                   z                 mag                  vx                  vy                  vz
     taud_Al1             0.02669            0.982777               3e-05                   0                   0                   0                   0
     taud_Al2            0.023514            0.483068            0.500002                   0                   0                   0                   0
     taud_Al3            0.522649            0.981899            0.499974                   0                   0                   0                   0
     taud_Al4            0.527148            0.482256            0.999994                   0                   0                   0                   0


 READING ORBITAL FILE NAMES FOR LCAO
 orbital file: Al_gga_10au_100Ry_3s3p2d.orb

                          Volume (Bohr^3) = 446.534
                             Volume (A^3) = 66.1695

 Lattice vectors: (Cartesian coordinate: in unit of a_0)
             +4.12959           -0.005768           -0.002402
             +0.00403            +3.87003           -0.001863
            -0.000266           -0.001379            +4.14035
 Reciprocal vectors: (Cartesian coordinate: in unit of 2 pi/a_0)
            +0.242155        -0.000252157        +1.54734e-05
         +0.000360964           +0.258396        +8.60855e-05
         +0.000140647        +0.000116122           +0.241526




 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Reading pseudopotentials files:                                    |
 | The pseudopotential file is in UPF format. The 'NC' indicates that |
 | the type of pseudopotential is 'norm conserving'. Functional of    |
 | exchange and correlation is decided by 4 given parameters in UPF   |
 | file.  We also read in the 'core correction' if there exists.      |
 | Also we can read the valence electrons number and the maximal      |
 | angular momentum used in this pseudopotential. We also read in the |
 | trail wave function, trail atomic density and local-pseudopotential|
 | on logrithmic grid. The non-local pseudopotential projector is also|
 | read in if there is any.                                           |
 |                                                                    |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




                PAO radial cut off (Bohr) = 15

 Read in pseudopotential file is Al.PD04.PBE.UPF
                     pseudopotential type = NC
          exchange-correlation functional = PBE
                 nonlocal core correction = 1
                        valence electrons = 3
                                     lmax = 2
                           number of zeta = 2
                     number of projectors = 6
                           L of projector = 0
                           L of projector = 0
                           L of projector = 1
                           L of projector = 1
                           L of projector = 2
                           L of projector = 2
     initial pseudo atomic orbital number = 16
                                   NLOCAL = 88




 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Setup plane waves of charge/potential:                             |
 | Use the energy cutoff and the lattice vectors to generate the      |
 | dimensions of FFT grid. The number of FFT grid on each processor   |
 | is 'nrxx'. The number of plane wave basis in reciprocal space is   |
 | different for charege/potential and wave functions. We also set    |
 | the 'sticks' for the parallel of FFT. The number of plane waves    |
 | is 'npw' in each processor.                                        |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





 SETUP THE PLANE WAVE BASIS
 energy cutoff for charge/potential (unit:Ry) = 80
          [fft grid for charge/potential] = 24, 24, 24
                      [fft grid division] = 2, 2, 2
      [big fft grid for charge/potential] = 12, 12, 12
                                     nbxx = 288
                                     nrxx = 2304

 SETUP PLANE WAVES FOR CHARGE/POTENTIAL
                    number of plane waves = 5393
                         number of sticks = 371

 PARALLEL PW FOR CHARGE/POTENTIAL
     PROC   COLUMNS(POT)             PW
        1             46            674
        2             46            674
        3             46            674
        4             46            674
        5             46            675
        6             47            674
        7             47            674
        8             47            674
 --------------- sum -------------------
        8            371           5393
                            number of |g| = 648
                                  max |g| = 7.21545
                                  min |g| = 0.125452

 SETUP THE ELECTRONS NUMBER
            electron number of element Al = 3
      total electron number of element Al = 12
                           occupied bands = 6
                                   NLOCAL = 88
                                   NBANDS = 16
                                   NBANDS = 16
 DONE : SETUP UNITCELL Time : 0.0599461 (SEC)





 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Setup K-points                                                     |
 | We setup the k-points according to input parameters.               |
 | The reduced k-points are set according to symmetry operations.     |
 | We treat the spin as another set of k-points.                      |
 |                                                                    |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





 SETUP K-POINTS
                                    nspin = 1
                   Input type of k points = Monkhorst-Pack(Gamma)
                                   nkstot = 64
                               nkstot_ibz = 36
      IBZ             DirectX             DirectY             DirectZ              Weight    ibz2bz
        1                   0                   0                   0            0.015625         0
        2                0.75                   0                   0             0.03125         1
        3                 0.5                   0                   0            0.015625         2
        4                   0                0.75                   0             0.03125         4
        5                0.75                0.75                   0             0.03125         5
        6                 0.5                0.75                   0             0.03125         6
        7                0.25                0.75                   0             0.03125         7
        8                   0                 0.5                   0            0.015625         8
        9                0.75                 0.5                   0             0.03125         9
       10                 0.5                 0.5                   0            0.015625        10
       11                   0                   0                0.75             0.03125        16
       12                0.75                   0                0.75             0.03125        17
       13                 0.5                   0                0.75             0.03125        18
       14                0.25                   0                0.75             0.03125        19
       15                   0                0.75                0.75             0.03125        20
       16                0.75                0.75                0.75             0.03125        21
       17                 0.5                0.75                0.75             0.03125        22
       18                0.25                0.75                0.75             0.03125        23
       19                   0                 0.5                0.75             0.03125        24
       20                0.75                 0.5                0.75             0.03125        25
       21                 0.5                 0.5                0.75             0.03125        26
       22                0.25                 0.5                0.75             0.03125        27
       23                   0                0.25                0.75             0.03125        28
       24                0.75                0.25                0.75             0.03125        29
       25                 0.5                0.25                0.75             0.03125        30
       26                0.25                0.25                0.75             0.03125        31
       27                   0                   0                 0.5            0.015625        32
       28                0.75                   0                 0.5             0.03125        33
       29                 0.5                   0                 0.5            0.015625        34
       30                   0                0.75                 0.5             0.03125        36
       31                0.75                0.75                 0.5             0.03125        37
       32                 0.5                0.75                 0.5             0.03125        38
       33                0.25                0.75                 0.5             0.03125        39
       34                   0                 0.5                 0.5            0.015625        40
       35                0.75                 0.5                 0.5             0.03125        41
       36                 0.5                 0.5                 0.5            0.015625        42
                               nkstot now = 36

  KPOINTS            DIRECT_X            DIRECT_Y            DIRECT_Z              WEIGHT
        1                   0                   0                   0            0.015625
        2                0.75                   0                   0             0.03125
        3                 0.5                   0                   0            0.015625
        4                   0                0.75                   0             0.03125
        5                0.75                0.75                   0             0.03125
        6                 0.5                0.75                   0             0.03125
        7                0.25                0.75                   0             0.03125
        8                   0                 0.5                   0            0.015625
        9                0.75                 0.5                   0             0.03125
       10                 0.5                 0.5                   0            0.015625
       11                   0                   0                0.75             0.03125
       12                0.75                   0                0.75             0.03125
       13                 0.5                   0                0.75             0.03125
       14                0.25                   0                0.75             0.03125
       15                   0                0.75                0.75             0.03125
       16                0.75                0.75                0.75             0.03125
       17                 0.5                0.75                0.75             0.03125
       18                0.25                0.75                0.75             0.03125
       19                   0                 0.5                0.75             0.03125
       20                0.75                 0.5                0.75             0.03125
       21                 0.5                 0.5                0.75             0.03125
       22                0.25                 0.5                0.75             0.03125
       23                   0                0.25                0.75             0.03125
       24                0.75                0.25                0.75             0.03125
       25                 0.5                0.25                0.75             0.03125
       26                0.25                0.25                0.75             0.03125
       27                   0                   0                 0.5            0.015625
       28                0.75                   0                 0.5             0.03125
       29                 0.5                   0                 0.5            0.015625
       30                   0                0.75                 0.5             0.03125
       31                0.75                0.75                 0.5             0.03125
       32                 0.5                0.75                 0.5             0.03125
       33                0.25                0.75                 0.5             0.03125
       34                   0                 0.5                 0.5            0.015625
       35                0.75                 0.5                 0.5             0.03125
       36                 0.5                 0.5                 0.5            0.015625

           k-point number in this process = 36
       minimum distributed K point number = 36

  KPOINTS         CARTESIAN_X         CARTESIAN_Y         CARTESIAN_Z              WEIGHT
        1                   0                   0                   0             0.03125
        2            0.181616        -0.000189118         1.16051e-05              0.0625
        3            0.121077        -0.000126079         7.73671e-06             0.03125
        4         0.000270723            0.193797         6.45641e-05              0.0625
        5            0.181887            0.193608         7.61692e-05              0.0625
        6            0.121348            0.193671         7.23008e-05              0.0625
        7           0.0608094            0.193734         6.84324e-05              0.0625
        8         0.000180482            0.129198         4.30427e-05             0.03125
        9            0.181796            0.129009         5.46478e-05              0.0625
       10            0.121258            0.129072         5.07794e-05             0.03125
       11         0.000105485         8.70915e-05            0.181144              0.0625
       12            0.181721        -0.000102026            0.181156              0.0625
       13            0.121183         -3.8987e-05            0.181152              0.0625
       14           0.0606441         2.40523e-05            0.181148              0.0625
       15         0.000376209            0.193884            0.181209              0.0625
       16            0.181992            0.193695             0.18122              0.0625
       17            0.121454            0.193758            0.181216              0.0625
       18           0.0609149            0.193821            0.181213              0.0625
       19         0.000285967            0.129285            0.181187              0.0625
       20            0.181902            0.129096            0.181199              0.0625
       21            0.121363            0.129159            0.181195              0.0625
       22           0.0608246            0.129222            0.181191              0.0625
       23         0.000195726           0.0646861            0.181166              0.0625
       24            0.181812           0.0644969            0.181177              0.0625
       25            0.121273             0.06456            0.181173              0.0625
       26           0.0607344            0.064623             0.18117              0.0625
       27         7.03235e-05          5.8061e-05            0.120763             0.03125
       28            0.181686        -0.000131057            0.120774              0.0625
       29            0.121148        -6.80175e-05             0.12077             0.03125
       30         0.000341047            0.193855            0.120827              0.0625
       31            0.181957            0.193666            0.120839              0.0625
       32            0.121418            0.193729            0.120835              0.0625
       33           0.0608797            0.193792            0.120831              0.0625
       34         0.000250806            0.129256            0.120806             0.03125
       35            0.181867            0.129067            0.120817              0.0625
       36            0.121328             0.12913            0.120814             0.03125

  KPOINTS            DIRECT_X            DIRECT_Y            DIRECT_Z              WEIGHT
        1                   0                   0                   0             0.03125
        2                0.75                   0                   0              0.0625
        3                 0.5                   0                   0             0.03125
        4                   0                0.75                   0              0.0625
        5                0.75                0.75                   0              0.0625
        6                 0.5                0.75                   0              0.0625
        7                0.25                0.75                   0              0.0625
        8                   0                 0.5                   0             0.03125
        9                0.75                 0.5                   0              0.0625
       10                 0.5                 0.5                   0             0.03125
       11                   0                   0                0.75              0.0625
       12                0.75                   0                0.75              0.0625
       13                 0.5                   0                0.75              0.0625
       14                0.25                   0                0.75              0.0625
       15                   0                0.75                0.75              0.0625
       16                0.75                0.75                0.75              0.0625
       17                 0.5                0.75                0.75              0.0625
       18                0.25                0.75                0.75              0.0625
       19                   0                 0.5                0.75              0.0625
       20                0.75                 0.5                0.75              0.0625
       21                 0.5                 0.5                0.75              0.0625
       22                0.25                 0.5                0.75              0.0625
       23                   0                0.25                0.75              0.0625
       24                0.75                0.25                0.75              0.0625
       25                 0.5                0.25                0.75              0.0625
       26                0.25                0.25                0.75              0.0625
       27                   0                   0                 0.5             0.03125
       28                0.75                   0                 0.5              0.0625
       29                 0.5                   0                 0.5             0.03125
       30                   0                0.75                 0.5              0.0625
       31                0.75                0.75                 0.5              0.0625
       32                 0.5                0.75                 0.5              0.0625
       33                0.25                0.75                 0.5              0.0625
       34                   0                 0.5                 0.5             0.03125
       35                0.75                 0.5                 0.5              0.0625
       36                 0.5                 0.5                 0.5             0.03125
 DONE : INIT K-POINTS Time : 0.0622621 (SEC)





 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Setup plane waves of wave functions:                               |
 | Use the energy cutoff and the lattice vectors to generate the      |
 | dimensions of FFT grid. The number of FFT grid on each processor   |
 | is 'nrxx'. The number of plane wave basis in reciprocal space is   |
 | different for charege/potential and wave functions. We also set    |
 | the 'sticks' for the parallel of FFT. The number of plane wave of  |
 | each k-point is 'npwk[ik]' in each processor                       |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





 SETUP PLANE WAVES FOR WAVE FUNCTIONS
     energy cutoff for wavefunc (unit:Ry) = 20
            [fft grid for wave functions] = 24, 24, 24
                    number of plane waves = 1309
                         number of sticks = 141

 PARALLEL PW FOR WAVE FUNCTIONS
     PROC   COLUMNS(POT)             PW
        1             17            163
        2             17            163
        3             17            163
        4             18            164
        5             18            164
        6             18            164
        7             18            164
        8             18            164
 --------------- sum -------------------
        8            141           1309
 DONE : INIT PLANEWAVE Time : 0.0637991 (SEC)

 DONE : INIT CHARGE Time : 0.0729671 (SEC)

                                 init_chg = atomic
 DONE : INIT POTENTIAL Time : 0.109215 (SEC)





 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Setup numerical orbitals:                                          |
 | This part setup: numerical atomic orbitals, non-local projectors   |
 | and neutral potential (1D). The atomic orbitals information        |
 | including the radius, angular momentum and zeta number.            |
 | The neutral potential is the sum of local part of pseudopotential  |
 | and potential given by atomic charge, they will cancel out beyond  |
 | a certain radius cutoff, because the Z/r character.                |
 |                                                                    |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





 SETUP ONE DIMENSIONAL ORBITALS/POTENTIAL
                        delta k  (1/Bohr) = 0.01
                        delta r    (Bohr) = 0.01
                        dr_uniform (Bohr) = 0.001
                        rmax       (Bohr) = 30
                                    kmesh = 451
      ORBITAL  L  N      nr      dr    RCUT  CHECK_UNIT    NEW_UNIT
            1  0  0    1001    0.01      10           1           1
            2  0  1    1001    0.01      10           1           1
            3  0  2    1001    0.01      10           1           1
            4  1  0    1001    0.01      10           1           1
            5  1  1    1001    0.01      10           1           1
            6  1  2    1001    0.01      10           1           1
            7  2  0    1001    0.01      10           1           1
            8  2  1    1001    0.01      10           1           1
 SET NONLOCAL PSEUDOPOTENTIAL PROJECTORS
 max number of nonlocal projetors among all species is 6

 SETUP THE TWO-CENTER INTEGRATION TABLES

 SETUP THE DIVISION OF H/S MATRIX
 divide the H&S matrix using 2D block algorithms.
                                     nb2d = 1
                  trace_loc_row dimension = 88
                  trace_loc_col dimension = 88
                                     nloc = 968

 -------------------------------------------
 SELF-CONSISTENT
 -------------------------------------------
    """

    return AbacusOutHeaderChunk(contents)


def test_header_n_atoms(header_chunk):
    assert header_chunk.n_atoms == 4


def test_header_n_bands(header_chunk):
    assert header_chunk.n_bands == 16


def test_header_n_occupied_bands(header_chunk):
    assert header_chunk.n_occupied_bands == 6


def test_header_lattice_constant(header_chunk):
    assert header_chunk.lattice_constant == 1


def test_header_n_spins(header_chunk):
    assert header_chunk.n_spins == 1


def test_header_initial_atoms(
    header_chunk, initial_cell, initial_positions, initial_velocities
):
    assert len(header_chunk.initial_atoms) == 4
    assert np.allclose(
        header_chunk.initial_atoms.cell.array,
        initial_cell,
    )
    assert np.allclose(
        header_chunk.initial_atoms.get_scaled_positions(), initial_positions
    )
    assert np.all(["Al"] * 4 == header_chunk.initial_atoms.symbols)
    assert np.allclose(header_chunk.initial_atoms.get_velocities(), initial_velocities)


def test_header_initial_cell(header_chunk, initial_cell):
    assert np.allclose(header_chunk.initial_cell, initial_cell)


def test_header_is_md(header_chunk):
    assert not header_chunk.is_md


def test_header_is_relaxation(header_chunk):
    assert not header_chunk.is_relaxation


def test_header_is_cell_relaxation(header_chunk):
    assert not header_chunk.is_cell_relaxation


def test_header_is_nscf(header_chunk):
    assert not header_chunk.is_nscf


def test_header_n_k_points(header_chunk):
    assert header_chunk.n_k_points == 36


@pytest.fixture
def k_points():
    return np.array(
        [
            [0, 0, 0],
            [0.75, 0, 0],
            [0.5, 0, 0],
            [0, 0.75, 0],
            [0.75, 0.75, 0],
            [0.5, 0.75, 0],
            [0.25, 0.75, 0],
            [0, 0.5, 0],
            [0.75, 0.5, 0],
            [0.5, 0.5, 0],
            [0, 0, 0.75],
            [0.75, 0, 0.75],
            [0.5, 0, 0.75],
            [0.25, 0, 0.75],
            [0, 0.75, 0.75],
            [0.75, 0.75, 0.75],
            [0.5, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.5, 0.75],
            [0.5, 0.5, 0.75],
            [0.25, 0.5, 0.75],
            [0, 0.25, 0.75],
            [0.75, 0.25, 0.75],
            [0.5, 0.25, 0.75],
            [0.25, 0.25, 0.75],
            [0, 0, 0.5],
            [0.75, 0, 0.5],
            [0.5, 0, 0.5],
            [0, 0.75, 0.5],
            [0.75, 0.75, 0.5],
            [0.5, 0.75, 0.5],
            [0.25, 0.75, 0.5],
            [0, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )


@pytest.fixture
def k_point_weights():
    return np.array(
        [
            0.03125,
            0.0625,
            0.03125,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.03125,
            0.0625,
            0.03125,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.03125,
            0.0625,
            0.03125,
            0.0625,
            0.0625,
            0.0625,
            0.0625,
            0.03125,
            0.0625,
            0.03125,
        ]
    )


def test_header_k_point_weights(header_chunk, k_point_weights):
    assert np.allclose(header_chunk.k_point_weights, k_point_weights)


def test_header_k_points(header_chunk, k_points):
    assert np.allclose(header_chunk.k_points, k_points)


def test_header_out_dir(header_chunk):
    assert header_chunk.out_dir == "OUT.ABACUS"


def test_header_header_summary(header_chunk, k_points, k_point_weights):
    header_summary = {
        "lattice_constant": 1,
        "initial_atoms": header_chunk.initial_atoms,
        "initial_cell": header_chunk.initial_cell,
        "is_nscf": False,
        "is_relaxation": False,
        "is_cell_relaxation": False,
        "is_md": False,
        "n_atoms": 4,
        "n_bands": 16,
        "n_occupied_bands": 6,
        "n_spins": 1,
        "n_k_points": 36,
        "k_points": k_points,
        "k_point_weights": k_point_weights,
        "out_dir": "OUT.ABACUS",
    }
    for key, val in header_chunk.header_summary.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, header_summary[key])
        else:
            assert val == header_summary[key]


@pytest.fixture
def calc_chunk(header_chunk):
    contents = """
-------------------------------------------
 SELF-CONSISTENT
 -------------------------------------------




 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 |                                                                    |
 | Search adjacent atoms:                                             |
 | Set the adjacent atoms for each atom and set the periodic boundary |
 | condition for the atoms on real space FFT grid. For k-dependent    |
 | algorithm, we also need to set the sparse H and S matrix element   |
 | for each atom.                                                     |
 |                                                                    |
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





 SETUP SEARCHING RADIUS FOR PROGRAM TO SEARCH ADJACENT ATOMS
                  longest orb rcut (Bohr) = 10
   longest nonlocal projector rcut (Bohr) = 1.78
              searching radius is (Bohr)) = 23.6
         searching radius unit is (Bohr)) = 1.89

 SETUP EXTENDED REAL SPACE GRID FOR GRID INTEGRATION
                        [real space grid] = 24, 24, 24
               [big cell numbers in grid] = 12, 12, 12
           [meshcell numbers in big cell] = 2, 2, 2
                      [extended fft grid] = 16, 17, 16
              [dimension of extened grid] = 45, 47, 45
                            UnitCellTotal = 125
              Atom number in sub-FFT-grid = 4
    Local orbitals number in sub-FFT-grid = 88
                                ParaV.nnr = 90750
                                     nnrg = 621456
                                nnrg_last = 0
                                 nnrg_now = 621456

 LCAO ALGORITHM --------------- ION=   1  ELEC=   1--------------------------------
 Memory of pvpR : 4.74 MB

 Density error is 0.0988016559521

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5193647766                -251.968884302
     E_Harris                -18.5253805788                 -252.05073349
      E_Fermi               +0.567072941646                  +7.715423188

 LCAO ALGORITHM --------------- ION=   1  ELEC=   2--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 0.027015947877

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191058244                -251.965361078
     E_Harris                -18.5195737462                -251.971727479
      E_Fermi               +0.565556173933                +7.69478650457

 LCAO ALGORITHM --------------- ION=   1  ELEC=   3--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 0.000925255846583

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347041                -251.965754005
     E_Harris                -18.5191348955                -251.965756609
      E_Fermi               +0.565241512553                +7.69050531687

 LCAO ALGORITHM --------------- ION=   1  ELEC=   4--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 0.000185664049322

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347284                -251.965754336
     E_Harris                -18.5191348877                -251.965756504
      E_Fermi               +0.565259938757                +7.69075601822

 LCAO ALGORITHM --------------- ION=   1  ELEC=   5--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 2.44751513351e-05

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347915                -251.965755195
     E_Harris                -18.5191347935                -251.965755222
      E_Fermi               +0.565260356432                +7.69076170099

 LCAO ALGORITHM --------------- ION=   1  ELEC=   6--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 4.27020212175e-06

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347923                -251.965755205
     E_Harris                -18.5191347924                -251.965755206
      E_Fermi               +0.565260393494                +7.69076220524

 LCAO ALGORITHM --------------- ION=   1  ELEC=   7--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 6.34504548203e-07

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347923                -251.965755205
     E_Harris                -18.5191347923                -251.965755205
      E_Fermi               +0.565260382585                +7.69076205682

 LCAO ALGORITHM --------------- ION=   1  ELEC=   8--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 9.91767704103e-08

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347924                -251.965755206
     E_Harris                -18.5191347924                -251.965755206
      E_Fermi                +0.56526038309                +7.69076206368

 LCAO ALGORITHM --------------- ION=   1  ELEC=   9--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 3.34835620614e-08

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347924                -251.965755206
     E_Harris                -18.5191347924                -251.965755206
      E_Fermi               +0.565260383012                +7.69076206262

 LCAO ALGORITHM --------------- ION=   1  ELEC=  10--------------------------------
 Memory of pvpR : 4.74133300781 MB

 Density error is 7.40371912424e-10

       Energy                       Rydberg                            eV
   E_KohnSham                -18.5191347923                -251.965755206
     E_Harris                -18.5191347923                -251.965755206
       E_band                +3.09357662253                 +42.090269266
   E_one_elec                +11.7057043728                +159.264278573
    E_Hartree              +0.0305562343346               +0.415738896374
         E_xc                -8.66110081062                -117.840321977
      E_Ewald                -21.5934730488                -293.794273073
      E_demet            -0.000821540072651              -0.0111776261234
      E_descf                            +0                            +0
     E_efield                            +0                            +0
        E_exx                            +0                            +0
      E_Fermi               +0.565260382998                +7.69076206244

 charge density convergence is achieved
 final etot is -251.965755206 eV

 STATE ENERGY(eV) AND OCCUPATIONS    NSPIN == 1
 1/36 kpoint (Cartesian) = 0 0 0 (83 pws)
       1       -3.20887      0.0312500
       2        4.70525      0.0312500
       3        4.73981      0.0312500
       4        5.72338      0.0312500
       5        5.83798      0.0312500
       6        5.89075      0.0312500
       7        7.23945      0.0312500
       8        12.7056        0.00000
       9        12.8371        0.00000
      10        12.8707        0.00000
      11        13.5199        0.00000
      12        13.5475        0.00000
      13        13.8596        0.00000
      14        13.9176        0.00000
      15        14.3333        0.00000
      16        14.3572        0.00000

 2/36 kpoint (Cartesian) = 0.181616 -0.000189118 1.16051e-05 (77 pws)
       1       -2.66803      0.0625000
       2        1.59681      0.0625000
       3        5.19723      0.0625000
       4        6.21242      0.0625000
       5        6.37064      0.0625000
       6        7.75054      0.0166992
       7        9.22099        0.00000
       8        10.0093        0.00000
       9        10.1952        0.00000
      10        10.6079        0.00000
      11        12.0343        0.00000
      12        14.1189        0.00000
      13        14.2955        0.00000
      14        14.8010        0.00000
      15        17.5104        0.00000
      16        17.9220        0.00000

 3/36 kpoint (Cartesian) = 0.121077 -0.000126079 7.73671e-06 (78 pws)
       1       -1.05595      0.0312500
       2       -1.05149      0.0312500
       3        6.69815      0.0312500
       4        6.70329      0.0312500
       5        7.69781      0.0147131
       6        7.70315      0.0140247
       7        7.86453     0.00110771
       8        7.86774     0.00102863
       9        9.59253        0.00000
      10        9.59569        0.00000
      11        15.6265        0.00000
      12        15.6284        0.00000
      13        15.6639        0.00000
      14        15.6682        0.00000
      15        15.9441        0.00000
      16        15.9503        0.00000

 4/36 kpoint (Cartesian) = 0.000270723 0.193797 6.45641e-05 (88 pws)
       1       -2.59268      0.0625000
       2        2.24456      0.0625000
       3        5.26960      0.0625000
       4        5.30397      0.0625000
       5        6.44363      0.0625000
       6        6.49623      0.0625000
       7        9.84671        0.00000
       8        9.87972        0.00000
       9        11.2326        0.00000
      10        11.2863        0.00000
      11        11.7976        0.00000
      12        13.3434        0.00000
      13        13.3747        0.00000
      14        13.3830        0.00000
      15        16.9466        0.00000
      16        17.5967        0.00000

 5/36 kpoint (Cartesian) = 0.181887 0.193608 7.61692e-05 (79 pws)
       1       -2.05263      0.0625000
       2        2.19224      0.0625000
       3        2.76626      0.0625000
       4        5.75950      0.0625000
       5        6.68305      0.0625000
       6        7.24888      0.0624999
       7        9.79410        0.00000
       8        10.3486        0.00000
       9        10.5241        0.00000
      10        11.0421        0.00000
      11        11.7923        0.00000
      12        12.4396        0.00000
      13        14.5069        0.00000
      14        14.9101        0.00000
      15        15.5199        0.00000
      16        17.0201        0.00000

 6/36 kpoint (Cartesian) = 0.121348 0.193671 7.23008e-05 (82 pws)
       1      -0.447792      0.0625000
       2      -0.443050      0.0625000
       3        4.32167      0.0625000
       4        4.32815      0.0625000
       5        7.26285      0.0624997
       6        7.26822      0.0624996
       7        8.61307        0.00000
       8        8.61741        0.00000
       9        11.8824        0.00000
      10        11.8894        0.00000
      11        13.1428        0.00000
      12        13.1497        0.00000
      13        13.8883        0.00000
      14        13.8971        0.00000
      15        16.2107        0.00000
      16        16.2155        0.00000

 7/36 kpoint (Cartesian) = 0.0608094 0.193734 6.84324e-05 (85 pws)
       1       -2.05346      0.0625000
       2        2.19458      0.0625000
       3        2.76858      0.0625000
       4        5.75881      0.0625000
       5        6.68061      0.0625000
       6        7.24337      0.0624999
       7        9.79628        0.00000
       8        10.3508        0.00000
       9        10.5216        0.00000
      10        11.0452        0.00000
      11        11.7934        0.00000
      12        12.4353        0.00000
      13        14.5000        0.00000
      14        14.9229        0.00000
      15        15.5143        0.00000
      16        17.0306        0.00000

 8/36 kpoint (Cartesian) = 0.000180482 0.129198 4.30427e-05 (84 pws)
       1      -0.761449      0.0312500
       2      -0.754931      0.0312500
       3        6.97534      0.0312500
       4        6.98316      0.0312500
       5        7.00921      0.0312500
       6        7.01702      0.0312500
       7        7.86440     0.00111085
       8        7.86475     0.00110206
       9        8.81930        0.00000
      10        8.81961        0.00000
      11        14.9409        0.00000
      12        14.9489        0.00000
      13        14.9718        0.00000
      14        14.9799        0.00000
      15        15.4646        0.00000
      16        15.4746        0.00000

 9/36 kpoint (Cartesian) = 0.181796 0.129009 5.46478e-05 (84 pws)
       1      -0.228286      0.0625000
       2      -0.221569      0.0625000
       3        3.96830      0.0625000
       4        3.97640      0.0625000
       5        7.46827      0.0618519
       6        7.47633      0.0616931
       7        8.82053        0.00000
       8        8.82625        0.00000
       9        11.5331        0.00000
      10        11.5426        0.00000
      11        12.3903        0.00000
      12        12.4005        0.00000
      13        12.8066        0.00000
      14        12.8147        0.00000
      15        18.7107        0.00000
      16        18.7162        0.00000

 10/36 kpoint (Cartesian) = 0.121258 0.129072 5.07794e-05 (84 pws)
       1        1.34882      0.0312500
       2        1.35413      0.0312500
       3        1.37329      0.0312500
       4        1.37881      0.0312500
       5        8.96147        0.00000
       6        8.96734        0.00000
       7        9.00504        0.00000
       8        9.01124        0.00000
       9        10.3700        0.00000
      10        10.3733        0.00000
      11        10.3772        0.00000
      12        10.3808        0.00000
      13        17.6119        0.00000
      14        17.6219        0.00000
      15        17.6447        0.00000
      16        17.6542        0.00000

 11/36 kpoint (Cartesian) = 0.000105485 8.70915e-05 0.181144 (80 pws)
       1       -2.67086      0.0625000
       2        1.57256      0.0625000
       3        5.22974      0.0625000
       4        6.20999      0.0625000
       5        6.42073      0.0625000
       6        7.74875      0.0170828
       7        9.23058        0.00000
       8        9.94318        0.00000
       9        10.1718        0.00000
      10        10.6355        0.00000
      11        12.0100        0.00000
      12        14.1438        0.00000
      13        14.3507        0.00000
      14        14.8285        0.00000
      15        17.4676        0.00000
      16        17.9566        0.00000

 12/36 kpoint (Cartesian) = 0.181721 -0.000102026 0.181156 (75 pws)
       1       -2.13008      0.0625000
       2        2.09488      0.0625000
       3        2.11697      0.0625000
       4        6.22547      0.0625000
       5        6.69537      0.0625000
       6        8.38296    1.95295e-14
       7        10.4258        0.00000
       8        10.4899        0.00000
       9        10.6633        0.00000
      10        10.6857        0.00000
      11        12.5442        0.00000
      12        12.5624        0.00000
      13        13.7551        0.00000
      14        14.7836        0.00000
      15        15.6123        0.00000
      16        16.4180        0.00000

 13/36 kpoint (Cartesian) = 0.121183 -3.89870e-05 0.181152 (80 pws)
       1      -0.524461      0.0625000
       2      -0.519219      0.0625000
       3        3.65760      0.0625000
       4        3.66665      0.0625000
       5        8.17871    1.23088e-08
       6        8.18457    8.92404e-09
       7        9.92238        0.00000
       8        9.92759        0.00000
       9        12.0371        0.00000
      10        12.0519        0.00000
      11        12.1840        0.00000
      12        12.1935        0.00000
      13        13.9636        0.00000
      14        13.9730        0.00000
      15        16.0983        0.00000
      16        16.1077        0.00000

 14/36 kpoint (Cartesian) = 0.0606441 2.40523e-05 0.181148 (82 pws)
       1       -2.13148      0.0625000
       2        2.09888      0.0625000
       3        2.12095      0.0625000
       4        6.21466      0.0625000
       5        6.69412      0.0625000
       6        8.38003    2.44180e-14
       7        10.4203        0.00000
       8        10.4844        0.00000
       9        10.6671        0.00000
      10        10.6894        0.00000
      11        12.5470        0.00000
      12        12.5653        0.00000
      13        13.7763        0.00000
      14        14.7714        0.00000
      15        15.6300        0.00000
      16        16.4077        0.00000

 15/36 kpoint (Cartesian) = 0.000376209 0.193884 0.181209 (88 pws)
       1       -2.05489      0.0625000
       2        2.16671      0.0625000
       3        2.76195      0.0625000
       4        5.79120      0.0625000
       5        6.68976      0.0625000
       6        7.27411      0.0624995
       7        9.80239        0.00000
       8        10.3755        0.00000
       9        10.4680        0.00000
      10        11.0668        0.00000
      11        11.8353        0.00000
      12        12.4409        0.00000
      13        14.5181        0.00000
      14        14.8428        0.00000
      15        15.5539        0.00000
      16        16.9854        0.00000

 16/36 kpoint (Cartesian) = 0.181992 0.193695 0.181220 (78 pws)
       1       -1.51568      0.0625000
       2        2.68673      0.0625000
       3        2.71028      0.0625000
       4        3.28062      0.0625000
       5        6.83610      0.0625000
       6        7.39503      0.0624340
       7        7.41019      0.0623893
       8        11.0843        0.00000
       9        11.1478        0.00000
      10        11.4749        0.00000
      11        12.8678        0.00000
      12        14.3099        0.00000
      13        15.3293        0.00000
      14        15.3954        0.00000
      15        16.2880        0.00000
      16        17.5648        0.00000

 17/36 kpoint (Cartesian) = 0.121454 0.193758 0.181216 (83 pws)
       1      0.0823322      0.0625000
       2      0.0885675      0.0625000
       3        4.22457      0.0625000
       4        4.23174      0.0625000
       5        4.86879      0.0625000
       6        4.87322      0.0625000
       7        8.92065        0.00000
       8        8.93428        0.00000
       9        12.5659        0.00000
      10        12.5815        0.00000
      11        14.4283        0.00000
      12        14.4397        0.00000
      13        16.6514        0.00000
      14        16.6639        0.00000
      15        16.7501        0.00000
      16        16.7577        0.00000

 18/36 kpoint (Cartesian) = 0.0609149 0.193821 0.181213 (86 pws)
       1       -1.51790      0.0625000
       2        2.68995      0.0625000
       3        2.71644      0.0625000
       4        3.28166      0.0625000
       5        6.82677      0.0625000
       6        7.40113      0.0624185
       7        7.40847      0.0623955
       8        11.0761        0.00000
       9        11.1369        0.00000
      10        11.4562        0.00000
      11        12.8622        0.00000
      12        14.3310        0.00000
      13        15.3263        0.00000
      14        15.4046        0.00000
      15        16.3052        0.00000
      16        17.5635        0.00000

 19/36 kpoint (Cartesian) = 0.000285967 0.129285 0.181187 (88 pws)
       1      -0.231337      0.0625000
       2      -0.223847      0.0625000
       3        3.94226      0.0625000
       4        3.95464      0.0625000
       5        7.49765      0.0611023
       6        7.50628      0.0607761
       7        8.86829        0.00000
       8        8.87549        0.00000
       9        11.5384        0.00000
      10        11.5519        0.00000
      11        12.3217        0.00000
      12        12.3413        0.00000
      13        12.8328        0.00000
      14        12.8452        0.00000
      15        18.6999        0.00000
      16        18.7117        0.00000

 20/36 kpoint (Cartesian) = 0.181902 0.129096 0.181199 (80 pws)
       1       0.300604      0.0625000
       2       0.308978      0.0625000
       3        4.35297      0.0625000
       4        4.35603      0.0625000
       5        4.60670      0.0625000
       6        4.61213      0.0625000
       7        8.57007        0.00000
       8        8.58715        0.00000
       9        12.6354        0.00000
      10        12.6404        0.00000
      11        13.0740        0.00000
      12        13.0778        0.00000
      13        15.9249        0.00000
      14        15.9345        0.00000
      15        18.1432        0.00000
      16        18.1534        0.00000

 21/36 kpoint (Cartesian) = 0.121363 0.129159 0.181195 (85 pws)
       1        1.87234      0.0625000
       2        1.87748      0.0625000
       3        1.89672      0.0625000
       4        1.90249      0.0625000
       5        6.00991      0.0625000
       6        6.01478      0.0625000
       7        6.03712      0.0625000
       8        6.04520      0.0625000
       9        14.2874        0.00000
      10        14.2946        0.00000
      11        14.3146        0.00000
      12        14.3276        0.00000
      13        18.0716        0.00000
      14        18.0816        0.00000
      15        18.1030        0.00000
      16        18.1135        0.00000

 22/36 kpoint (Cartesian) = 0.0608246 0.129222 0.181191 (85 pws)
       1       0.300006      0.0625000
       2       0.306846      0.0625000
       3        4.35335      0.0625000
       4        4.35951      0.0625000
       5        4.60853      0.0625000
       6        4.62280      0.0625000
       7        8.56222        0.00000
       8        8.57158        0.00000
       9        12.6259        0.00000
      10        12.6319        0.00000
      11        13.0642        0.00000
      12        13.0781        0.00000
      13        15.9462        0.00000
      14        15.9549        0.00000
      15        18.1620        0.00000
      16        18.1688        0.00000

 23/36 kpoint (Cartesian) = 0.000195726 0.0646861 0.181166 (84 pws)
       1       -2.05679      0.0625000
       2        2.17206      0.0625000
       3        2.76722      0.0625000
       4        5.78951      0.0625000
       5        6.68339      0.0625000
       6        7.26236      0.0624997
       7        9.80761        0.00000
       8        10.3807        0.00000
       9        10.4620        0.00000
      10        11.0735        0.00000
      11        11.8375        0.00000
      12        12.4303        0.00000
      13        14.5017        0.00000
      14        14.8710        0.00000
      15        15.5405        0.00000
      16        17.0094        0.00000

 24/36 kpoint (Cartesian) = 0.181812 0.0644969 0.181177 (79 pws)
       1       -1.51839      0.0625000
       2        2.69119      0.0625000
       3        2.71084      0.0625000
       4        3.28813      0.0625000
       5        6.84420      0.0625000
       6        7.38243      0.0624578
       7        7.40894      0.0623938
       8        11.0739        0.00000
       9        11.1411        0.00000
      10        11.4523        0.00000
      11        12.8550        0.00000
      12        14.3083        0.00000
      13        15.3587        0.00000
      14        15.4113        0.00000
      15        16.2853        0.00000
      16        17.5855        0.00000

 25/36 kpoint (Cartesian) = 0.121273 0.0645600 0.181173 (82 pws)
       1      0.0813117      0.0625000
       2      0.0859095      0.0625000
       3        4.22790      0.0625000
       4        4.23694      0.0625000
       5        4.87275      0.0625000
       6        4.88263      0.0625000
       7        8.90896        0.00000
       8        8.91481        0.00000
       9        12.5582        0.00000
      10        12.5701        0.00000
      11        14.4221        0.00000
      12        14.4296        0.00000
      13        16.6523        0.00000
      14        16.6595        0.00000
      15        16.7764        0.00000
      16        16.7862        0.00000

 26/36 kpoint (Cartesian) = 0.0607344 0.0646230 0.181170 (84 pws)
       1       -1.51895      0.0625000
       2        2.69587      0.0625000
       3        2.71258      0.0625000
       4        3.28455      0.0625000
       5        6.83004      0.0625000
       6        7.38424      0.0624549
       7        7.41986      0.0623480
       8        11.0683        0.00000
       9        11.1381        0.00000
      10        11.4475        0.00000
      11        12.8570        0.00000
      12        14.3294        0.00000
      13        15.3532        0.00000
      14        15.3938        0.00000
      15        16.3043        0.00000
      16        17.5897        0.00000

 27/36 kpoint (Cartesian) = 7.03235e-05 5.80610e-05 0.120763 (80 pws)
       1       -1.06503      0.0312500
       2       -1.06453      0.0312500
       3        6.72408      0.0312500
       4        6.72441      0.0312500
       5        7.69004      0.0157180
       6        7.69064      0.0156406
       7        7.90499    0.000405715
       8        7.90508    0.000404748
       9        9.58663        0.00000
      10        9.58693        0.00000
      11        15.5312        0.00000
      12        15.5318        0.00000
      13        15.7090        0.00000
      14        15.7112        0.00000
      15        15.9622        0.00000
      16        15.9647        0.00000

 28/36 kpoint (Cartesian) = 0.181686 -0.000131057 0.120774 (74 pws)
       1      -0.531487      0.0625000
       2      -0.528691      0.0625000
       3        3.67174      0.0625000
       4        3.67958      0.0625000
       5        8.17297    1.68163e-08
       6        8.17555    1.46191e-08
       7        9.91601        0.00000
       8        9.91986        0.00000
       9        12.0926        0.00000
      10        12.1067        0.00000
      11        12.1980        0.00000
      12        12.2058        0.00000
      13        13.9755        0.00000
      14        13.9837        0.00000
      15        16.0055        0.00000
      16        16.0147        0.00000

 29/36 kpoint (Cartesian) = 0.121148 -6.80175e-05 0.120770 (74 pws)
       1        1.05680      0.0312500
       2        1.06183      0.0312500
       3        1.06388      0.0312500
       4        1.06882      0.0312500
       5        9.66207        0.00000
       6        9.66956        0.00000
       7        9.67073        0.00000
       8        9.67806        0.00000
       9        11.5161        0.00000
      10        11.5175        0.00000
      11        11.5235        0.00000
      12        11.5249        0.00000
      13        16.5521        0.00000
      14        16.5589        0.00000
      15        16.5691        0.00000
      16        16.5760        0.00000

 30/36 kpoint (Cartesian) = 0.000341047 0.193855 0.120827 (89 pws)
       1      -0.458437      0.0625000
       2      -0.454712      0.0625000
       3        4.30902      0.0625000
       4        4.31934      0.0625000
       5        7.28620      0.0624992
       6        7.28965      0.0624990
       7        8.65500        0.00000
       8        8.65991        0.00000
       9        11.9012        0.00000
      10        11.9117        0.00000
      11        13.1850        0.00000
      12        13.1958        0.00000
      13        13.8688        0.00000
      14        13.8875        0.00000
      15        16.1114        0.00000
      16        16.1228        0.00000

 31/36 kpoint (Cartesian) = 0.181957 0.193666 0.120839 (78 pws)
       1      0.0734332      0.0625000
       2      0.0798648      0.0625000
       3        4.23925      0.0625000
       4        4.24335      0.0625000
       5        4.86209      0.0625000
       6        4.86975      0.0625000
       7        8.92633        0.00000
       8        8.94507        0.00000
       9        12.6208        0.00000
      10        12.6383        0.00000
      11        14.4118        0.00000
      12        14.4329        0.00000
      13        16.5579        0.00000
      14        16.5770        0.00000
      15        16.8063        0.00000
      16        16.8108        0.00000

 32/36 kpoint (Cartesian) = 0.121418 0.193729 0.120835 (86 pws)
       1        1.65533      0.0625000
       2        1.66069      0.0625000
       3        1.66188      0.0625000
       4        1.66856      0.0625000
       5        6.38129      0.0625000
       6        6.38610      0.0625000
       7        6.39066      0.0625000
       8        6.40128      0.0625000
       9        15.8945        0.00000
      10        15.8998        0.00000
      11        15.9091        0.00000
      12        15.9231        0.00000
      13        17.1047        0.00000
      14        17.1128        0.00000
      15        17.1220        0.00000
      16        17.1297        0.00000

 33/36 kpoint (Cartesian) = 0.0608797 0.193792 0.120831 (88 pws)
       1      0.0752943      0.0625000
       2      0.0763587      0.0625000
       3        4.23823      0.0625000
       4        4.24788      0.0625000
       5        4.86120      0.0625000
       6        4.87664      0.0625000
       7        8.92742        0.00000
       8        8.93021        0.00000
       9        12.6214        0.00000
      10        12.6300        0.00000
      11        14.4105        0.00000
      12        14.4272        0.00000
      13        16.5654        0.00000
      14        16.5670        0.00000
      15        16.8163        0.00000
      16        16.8280        0.00000

 34/36 kpoint (Cartesian) = 0.000250806 0.129256 0.120806 (90 pws)
       1        1.34544      0.0312500
       2        1.35016      0.0312500
       3        1.35502      0.0312500
       4        1.36007      0.0312500
       5        8.99683        0.00000
       6        9.00678        0.00000
       7        9.00763        0.00000
       8        9.01738        0.00000
       9        10.4146        0.00000
      10        10.4148        0.00000
      11        10.4244        0.00000
      12        10.4245        0.00000
      13        17.5360        0.00000
      14        17.5408        0.00000
      15        17.5593        0.00000
      16        17.5634        0.00000

 35/36 kpoint (Cartesian) = 0.181867 0.129067 0.120817 (81 pws)
       1        1.87159      0.0625000
       2        1.87652      0.0625000
       3        1.88072      0.0625000
       4        1.88695      0.0625000
       5        6.03035      0.0625000
       6        6.03686      0.0625000
       7        6.03874      0.0625000
       8        6.05075      0.0625000
       9        14.3536        0.00000
      10        14.3606        0.00000
      11        14.3644        0.00000
      12        14.3803        0.00000
      13        17.9979        0.00000
      14        18.0055        0.00000
      15        18.0200        0.00000
      16        18.0293        0.00000

 36/36 kpoint (Cartesian) = 0.121328 0.129130 0.120814 (82 pws)
       1        3.35129      0.0312500
       2        3.35310      0.0312500
       3        3.35414      0.0312500
       4        3.35900      0.0312500
       5        3.54771      0.0312500
       6        3.55102      0.0312500
       7        3.55377      0.0312500
       8        3.56544      0.0312500
       9        18.7019        0.00000
      10        18.7047        0.00000
      11        18.7191        0.00000
      12        18.7222        0.00000
      13        18.8552        0.00000
      14        18.8697        0.00000
      15        18.8750        0.00000
      16        18.8820        0.00000

 EFERMI = 7.690762062437759 eV
 OUT.ABACUS/ final etot is -251.9657552059826 eV
 correction force for each atom along direction 1 is -0.000394451
 correction force for each atom along direction 2 is -1.01249e-05
 correction force for each atom along direction 3 is -1.25861e-05

 ><><><><><><><><><><><><><><><><><><><><><><

    TOTAL-FORCE (eV/Angstrom)

 ><><><><><><><><><><><><><><><><><><><><><><

     atom              x              y              z
      Al1   -0.020115777  +0.0016108950 -0.00082046046
      Al2   +0.011204541   -0.015487210 +0.00048774547
      Al3   +0.035036644   +0.013664429 +0.00077866599
      Al4   -0.026125409 +0.00021188580 -0.00044595099


 ><><><><><><><><><><><><><><><><><><><><><><

 TOTAL-STRESS (KBAR)

 ><><><><><><><><><><><><><><><><><><><><><><

       +1.624103      +1.340990      +0.501469
       +1.340990      +1.115118      +0.447678
       +0.501469      +0.447678      -3.447143
 TOTAL-PRESSURE: -0.235974 KBAR


 --------------------------------------------
 !FINAL_ETOT_IS -251.9657552059825889 eV
 --------------------------------------------






  |CLASS_NAME---------|NAME---------------|TIME(Sec)-----|CALLS----|AVG------|PER%-------
                                      total      +30.86814        11     +2.81   +100.00%
               Run_lcao           lcao_line      +30.84332         1    +30.84    +99.92%
            ORB_control      read_orb_first       +0.16985         1     +0.17     +0.55%
          LCAO_Orbitals       Read_Orbitals       +0.16985         1     +0.17     +0.55%
            NOrbital_Lm       extra_uniform       +0.12679         8     +0.02     +0.41%
          Mathzone_Add1       Uni_Deriv_Phi       +0.12246         8     +0.02     +0.40%
            ORB_control      set_orb_tables       +1.06034         1     +1.06     +3.44%
         ORB_gen_tables          gen_tables       +1.06033         1     +1.06     +3.44%
          ORB_table_phi          init_Table       +0.67736         1     +0.68     +2.19%
          ORB_table_phi      cal_ST_Phi12_R       +0.66458       186     +0.00     +2.15%
         ORB_table_beta     init_Table_Beta       +0.24629         1     +0.25     +0.80%
         ORB_table_beta       VNL_PhiBeta_R       +0.24339        72     +0.00     +0.79%
              LOOP_ions            opt_ions      +29.51620         1    +29.52    +95.62%
        ESolver_KS_LCAO                 Run      +25.28325         1    +25.28    +81.91%
        ESolver_KS_LCAO           beforescf       +0.39912         1     +0.40     +1.29%
        ESolver_KS_LCAO        beforesolver       +0.38728         1     +0.39     +1.25%
            LCAO_Hamilt   set_lcao_matrices       +0.36804         1     +0.37     +1.19%
        LCAO_gen_fixedHbuild_Nonlocal_mu_new       +1.03465         2     +0.52     +3.35%
            HSolverLCAO               solve      +24.72715        10     +2.47    +80.11%
             HamiltLCAO              each_k      +15.84544       360     +0.04    +51.33%
         Gint_interface            cal_gint      +16.84423        21     +0.80    +54.57%
             HamiltLCAO                 H_k       +3.75565       180     +0.02    +12.17%
            LCAO_Hamilt        calculate_Hk       +3.13217       360     +0.01    +10.15%
                 Gint_k        folding_vl_k       +1.77828       360     +0.00     +5.76%
                 Gint_k              Distri       +1.31043       360     +0.00     +4.25%
               LCAO_nnr      folding_fixedH       +1.35026       360     +0.00     +4.37%
             Efficience                 H_k       +2.67913       180     +0.01     +8.68%
              DiagoElpa          elpa_solve       +3.25660       360     +0.01    +10.55%
          ElecStateLCAO            psiToRho       +2.47732        10     +0.25     +8.03%
            LCAO_Charge            cal_dk_k       +1.49368        10     +0.15     +4.84%
              LOOP_ions        force_stress       +4.22940         1     +4.23    +13.70%
      Force_Stress_LCAO      getForceStress       +4.22930         1     +4.23    +13.70%
           Force_LCAO_k            ftable_k       +4.16951         1     +4.17    +13.51%
           Force_LCAO_k          allocate_k       +1.15902         1     +1.16     +3.75%
           Force_LCAO_k      cal_foverlap_k       +0.11570         1     +0.12     +0.37%
           Force_LCAO_k          cal_edm_2d       +0.11134         1     +0.11     +0.36%
           Force_LCAO_k      cal_fvl_dphi_k       +0.22819         1     +0.23     +0.74%
           Force_LCAO_kcal_fvnl_dbeta_k_new       +2.62047         1     +2.62     +8.49%
 ----------------------------------------------------------------------------------------

 CLASS_NAME---------|NAME---------------|MEMORY(MB)--------
                                         +1587.3832
        ORB_table_phi               Jl(x)       +62.0387
  LocalOrbital_Charge      Density_Matrix        +4.7413
        allocate_pvpR        pvpR_reduced        +4.7413
            stress_lo                 dSR        +4.1542
        ORB_table_phi         Table_SR&TR        +3.9160
             force_lo                  dS        +2.0771
             force_lo               dTVNL        +2.0771
 ----------------------------------------------------------

 Start  Time  : Thu Jul 28 14:26:15 2022
 Finish Time  : Thu Jul 28 14:26:46 2022
 Total  Time  : 0 h 0 mins 31 secs

    """
    return AbacusOutCalcChunk(contents, header_chunk)


def test_calc_atoms(calc_chunk, initial_cell, initial_positions):
    assert len(calc_chunk.atoms) == 4
    assert np.allclose(calc_chunk.atoms.cell, initial_cell)
    assert np.allclose(calc_chunk.atoms.get_scaled_positions(), initial_positions)
    assert np.all(["Al"] * 4 == calc_chunk.atoms.symbols)


def test_calc_free_energy(calc_chunk):
    free_energy = -251.965755206
    assert np.abs(calc_chunk.free_energy - free_energy) < eps_hp
    assert (
        np.abs(calc_chunk.atoms.calc.get_property("free_energy") - free_energy) < eps_hp
    )
    assert np.abs(calc_chunk.results["free_energy"] - free_energy) < eps_hp


def test_calc_energy(calc_chunk):
    energy = -251.965755206
    assert np.abs(calc_chunk.energy - energy) < eps_hp
    assert np.abs(calc_chunk.atoms.get_potential_energy() - energy) < eps_hp
    assert np.abs(calc_chunk.results["energy"] - energy) < eps_hp


def test_calc_n_iter(calc_chunk):
    n_iter = 10
    assert calc_chunk.n_iter == n_iter
    assert calc_chunk.results["n_iter"] == n_iter


def test_calc_fermi_energy(calc_chunk):
    Ef = 7.690762062437759
    assert np.abs(calc_chunk.E_f - Ef) < eps_lp
    assert np.abs(calc_chunk.results["fermi_level"] - Ef) < eps_lp


def test_calc_converged(calc_chunk):
    assert calc_chunk.converged


@pytest.fixture
def eigenvalues_occupations():
    eigenvalues_occupancies = np.array(
        [
            [3.35129, 0.0312500],
            [3.35310, 0.0312500],
            [3.35414, 0.0312500],
            [3.35900, 0.0312500],
            [3.54771, 0.0312500],
            [3.55102, 0.0312500],
            [3.55377, 0.0312500],
            [3.56544, 0.0312500],
            [18.7019, 0.00000],
            [18.7047, 0.00000],
            [18.7191, 0.00000],
            [18.7222, 0.00000],
            [18.8552, 0.00000],
            [18.8697, 0.00000],
            [18.8750, 0.00000],
            [18.8820, 0.00000],
        ]
    )

    return eigenvalues_occupancies


def test_calc_eigenvalues(calc_chunk, eigenvalues_occupations):
    assert np.allclose(calc_chunk.eigenvalues[0][-1], eigenvalues_occupations[:, 0])
    assert np.allclose(
        calc_chunk.results["eigenvalues"][0][-1], eigenvalues_occupations[:, 0]
    )


def test_calc_ooccupations(calc_chunk, eigenvalues_occupations):
    assert np.allclose(calc_chunk.occupations[0][-1], eigenvalues_occupations[:, 1])
    assert np.allclose(
        calc_chunk.results["occupations"][0][-1], eigenvalues_occupations[:, 1]
    )


def test_calc_forces(calc_chunk):
    forces = np.array(
        [
            [-0.020115777, +0.0016108950, -0.00082046046],
            [+0.011204541, -0.015487210, +0.00048774547],
            [+0.035036644, +0.013664429, +0.00077866599],
            [-0.026125409, +0.00021188580, -0.00044595099],
        ]
    )
    assert np.allclose(calc_chunk.forces, forces)

    assert np.allclose(calc_chunk.atoms.get_forces(), forces)
    assert np.allclose(calc_chunk.results["forces"], forces)


def test_calc_stress(calc_chunk):
    stress = full_3x3_to_voigt_6_stress(
        np.array(
            [
                [1.624103, 1.340990, 0.501469],
                [1.340990, 1.115118, 0.447678],
                [0.501469, 0.447678, -3.447143],
            ]
        )
        * -0.1
        * GPa
    )
    assert np.allclose(calc_chunk.stress, stress)
    assert np.allclose(calc_chunk.atoms.get_stress(), stress)
    assert np.allclose(calc_chunk.results["stress"], stress)
