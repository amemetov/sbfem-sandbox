import math
import numpy as np
import sbfem.sbfem as sbfem


def example3SElements():
    """
    Original name: Exmpl3SElements (p.90)
    :return:
    """
    # Mesh
    # nodal coordinates. One node per row [x y]
    coord = np.array([[0, 0], [0, 1], [0, 3], [1, 0], [1, 1], [2, 0], [2, 1], [2, 3]])

    # Input S-element connectivity as a cell array (One S-element per cell).
    # In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2].
    sdConn = np.array([
        np.array([[1, 4], [4, 6], [6, 7], [7, 2], [2, 1]]),  # S-element 1
        np.array([[0, 3], [3, 4], [4, 1], [1, 0]]),  # S-element 2
        np.array([[3, 5], [5, 6], [6, 4], [4, 3]])  # S-element 3
    ])

    # coordinates of scaling centres of S-elements.
    sdSC = np.array([[1, 2], [0.5, 0.5], [1.5, 0.5]])  # one S-element per row

    # elascity matrix(plane stress).
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10E6, 0.25), den=2)

    # Boundary conditions
    # nodal forces. One force component per row: [Node Dir F]
    BC_Frc = np.array([[2, 2, 1E3], [7, 2, 1E3]])  # forces in KN
    # assemblage external forces
    ndn = 2  # 2 DOFs per node
    NDof = ndn * coord.shape[0]  # number of DOFs
    F = np.zeros((NDof))  # initializing right-hand side of equation [K]{u} = {F}
    F = sbfem.addNodalForces(BC_Frc, F)  # add prescribed nodal forces
    # displacement constraints. One constraint per row: [Node Dir Disp]
    BC_Disp = np.array([[0, 2, 0], [3, 1, 0], [3, 2, 0], [5, 2, 0]])

    return {'coord': coord,
            'sdConn': sdConn,
            'sdSC': sdSC,
            'mat': mat,
            'F': F,
            'BC_Disp': BC_Disp
            }


def example_3_1():
    """
    Example 3.1 Compute the solution of the scaled boundary finite element
    equation for the square S-element in Example 2.4 on page 65.
    Assume the Young’s modulus is equal to E = 10 GPa and Poisson’s ratio ν = 0.
    """
    # Solution of a square S-element
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    conn = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])  # [1:4; 2:4 1]’
    # elascity matrix(plane stress).
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10, 0), den=2)
    E0, E1, E2, M0 = sbfem.coeffMatricesOfSElement(xy, conn, mat)
    K, d, v, M = sbfem.sbfem(E0, E1, E2, M0)

    return {
        'in': {'xy': xy, 'conn': conn, 'mat': mat},
        'out': {'E0': E0, 'E1': E1, 'E2': E2, 'M0': M0,
                'K': K, 'd': d, 'v': v, 'M': M}
    }


def example_3_2():
    """
    Example 3.2 A regular pentagon S-element is shown in Figure 3.1.
    The vertices are on a circle with a radius of 1 m.
    Each edge is modelled with a 2-node line element.
    The nodal numbers and the line element numbers (in circle) are shown in Figure 3.1.
    The elasticity constants are: Young’s modulus E = 10 GPa and Poison’s ratio ν = 0.
    Consider plane stress states. Use the eigenvalue method to determine the solution of
    the scaled boundary finite element equation of the S-element.
    """
    # Solution of pentagon S-element
    radians = np.deg2rad([(-126 + i*72) for i in range(5)])
    xy = np.array([[math.cos(d), math.sin(d)] for d in radians])
    conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])  # [1:5; 2:5 1]'

    # elascity matrix(plane stress).
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10, 0), den=2)
    E0, E1, E2, M0 = sbfem.coeffMatricesOfSElement(xy, conn, mat)
    K, d, v, M = sbfem.sbfem(E0, E1, E2, M0)

    return {
        'in': {'radians': radians, 'xy': xy, 'conn': conn, 'mat': mat},
        'out': {'E0': E0, 'E1': E1, 'E2': E2, 'M0': M0,
                'K': K, 'd': d, 'v': v, 'M': M}
    }


def example_3_3():
    """
    Example 3.3 Rectangular Body Under Uniaxial Tension: Assembly of Global Equations and Solution
    A rectangular body is modelled by 3 S-elements as shown in Figure 3.3a.
    The dimensions (Unit: m) are indicated in Figure 3.3 with b = 1.
    The material constant are Young’s modulus E = 10 GPa and Poisson’s ratio ν = 0.25.
    The input of the S-element data and the assembly of the global stiffness matrix are illustrated.
    """
    example = example3SElements()
    coord = example['coord']
    sdConn = example['sdConn']
    sdSC = example['sdSC']
    mat = example['mat']

    # solution of S-elements and assemblage of global stiffness and mass matrices
    sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)

    return {
        'in': example,
        'out': {'sdSln': sdSln, 'K': K, 'M': M}
    }


def example_3_4():
    """
    Example 3.4 Uniaxial Tension of a Rectangular Body.
    Consider the problem shown in Figure 3.3, Example 3.3 on page 90.
    A vertical force F = 1000 KN∕m is applied at Nodes 3 and 8.
    Determine the nodal displacements and the support reactions.
    """
    example = example3SElements()
    coord = example['coord']
    sdConn = example['sdConn']
    sdSC = example['sdSC']
    mat = example['mat']
    F = example['F']
    BC_Disp = example['BC_Disp']

    # solution of S-elements and assemblage of global stiffness and mass matrices
    sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)

    # Static solution of nodal displacements and forces
    d, F = sbfem.solverStatics(K, BC_Disp, F)

    return {
        'in': example,
        'out': {'d': d, 'F': F}
    }
