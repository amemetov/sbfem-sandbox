import math
import numpy as np
import sbfem.sbfem as sbfem
import sbfem.utils as utils


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
        'out': {'d': d, 'F': F, 'sdSln': sdSln}
    }


def example_3_5():
    """
    Example 3.5 A Deep Cantilever Beam Subject to Bending.
    A deep cantilever beam of height H = 1 m and length L = 2 m is shown in Figure 3.6a.
    A bending moment M = 100 kNm, which is equivalent to the linearly distributed surface traction
    with p = 600kN∕m, is applied at the free end.
    The material properties are Young’s modulus E = 10 GPa and Poisson’s ratio ν = 0.2.
    Plane stress conditions are assumed. Determine the deflection of the beam.
    """

    # Mesh
    # nodal coordinates. One node per row [x y]
    coord = np.array([
        [0, 1], [0, 0.5], [0, 0], [0.68, 1], [0.68, 0.63], [0.49, 0.48],  [0.5, 0], [1.35, 1],
        [2, 1], [1, 0.45], [1.35, 0.62], [1, 0], [1.5, 0.47], [2, 0.5], [1.5, 0], [2, 0]
    ])

    # nodes of a polygon. The sequence follows counter-clockwise direction.
    polygon = utils.matlabToPythonIndices([
        np.array([3, 7, 6, 2]),
        np.array([15, 16, 14, 13]),
        np.array([2, 6, 5, 4, 1]),
        np.array([12, 15, 13, 11, 10]),
        np.array([7, 12, 10, 5, 6]),
        np.array([4, 5, 10, 11, 8]),
        np.array([8, 11, 13, 14, 9])
    ])

    # Input S-element connectivity as a cell array (One S-element per cell).
    # In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2].
    nsd = len(polygon)  # ltx number of S-elements
    sdConn = [None] * nsd  # initialising connectivity
    sdSC = np.zeros((nsd, 2))  # scaling centre
    for isub in range(nsd):
        # build connectivity
        sdConn[isub] = np.vstack((polygon[isub], np.hstack((polygon[isub][1:], polygon[isub][0:1])))).T
        # scaling centre at centroid of nodes (averages of nodal coorindates)
        sdSC[isub, :] = np.mean(coord[polygon[isub], :], axis=0)

    # Materials: elasticity matrix for plane stress condition
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10E9, 0.2), den=2000)  # mass density in kg∕m 3

    # Boundary conditions
    # displacement constraints (or prescribed acceleration in a response history analysis).
    # One constraint per row: [Node Dir Disp]
    BC_Disp = np.array([[0, 1, 0], [0, 2, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0], [2, 2, 0]])

    # assemble nodal forces
    ndn = 2  # 2 DOFs per node
    NDof = ndn*coord.shape[0]  # number of DOFs
    F = np.zeros(NDof)  # initializing right-hand side of equation [K]{u} = {F}
    edge = utils.matlabToPythonIndices(np.array([[16, 14], [14, 9]]))  # edges subject to tractions, one row per edge
    trac = np.array([[6E5, 0, 0, 0], [0, 0, -6E5, 0]]).T  # tractions, one column per edge
    F = sbfem.addSurfaceTraction(coord, edge, trac, F)

    # solution of S-elements and assemblage of global stiffness and mass matrices
    sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)

    # Static solution of nodal displacements and forces
    d, F = sbfem.solverStatics(K, BC_Disp, F)

    return {
        'in': {'coord': coord, 'sdConn': sdConn, 'sdSC': sdSC, 'mat': mat, 'F': F, 'BC_Disp': BC_Disp },
        'out': {'d': d, 'F': F}
    }


def example_3_6():
    """
    Example 3.6 An Edge-cracked Rectangular Body Subject to Tension
    An edge-cracked rectangular body is shown in Figure 3.8a with the length b = 0.1 m.
    The base of the body is fixed and a uniform tension p = 1 MPa is applied at the top.
    The material properties are Young’s modulus E = 10 GPa and Poisson’s ratio ν = 0.25.
    Plane stress conditions are assumed.
    Determine the crack opening displacements (CODs).
    """
    # Mesh
    # nodal coordinates in mm. One node per row [x y]
    b = 0.1
    coord = b*np.array([[0, 0], [0, -0.5], [0, -1], [0.5, -1], [1, -1], [1.5, -1], [2, -1],
                        [2, -0.5], [2, 0], [2, 0.5], [2, 1], [1.5, 1], [1, 1], [0.5, 1], [0, 1], [0, 0.5],
                        [0, 1E-14], [0, -2], [0, -3], [1, -3], [2, -3], [2, -2], [3, -3], [4, -3],
                        [4, -2], [4, -1], [3, -1], [4, 0], [4, 1], [3, 1], [4, 2], [4, 3], [3, 3],
                        [2, 3], [2, 2], [1, 3], [0, 3], [0, 2]])

    # Input S-element connectivity as a cell array (One S-element per cell).
    # In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2]
    sdConn = [
        utils.matlabToPythonIndices(np.vstack((np.arange(1, 17), np.arange(2, 18))).T),  # S-element 1
        utils.matlabToPythonIndices(np.array([[3, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 7], [4, 3], [5, 4], [6, 5], [7, 6]])),  # S-element 2
        utils.matlabToPythonIndices(np.array([[21, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 7], [22, 21], [7, 22]])),  # S-element 3
        utils.matlabToPythonIndices(np.array([[8, 7], [9, 8], [27, 26], [7, 27], [26, 28], [28, 29], [29, 30], [30, 11], [10, 9], [11, 10]])),  # S-element 4
        utils.matlabToPythonIndices(np.array([[30, 29], [11, 30], [29, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 11]])),  # S-element 5
        utils.matlabToPythonIndices(np.array([[12, 11], [13, 12], [14, 13], [15, 14], [35, 34], [11, 35], [34, 36], [36, 37], [37, 38], [38, 15]]))  # S-element 6
        ]
    # coordinates of scaling centres of S-elements, one S-element per row
    sdSC = b * np.array([[1, 0], [1, -2], [3, -2], [3, 0], [3, 2], [1, 2]])

    # Materials
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10E9, 0.25), den=2000)  # E in Pa, mass density in kg∕m 3

    # Boundary conditions
    # displacement constraints. One constraint per row: [Node Dir Disp]

    BC_Disp = np.array([[19, 1, 0], [19, 2, 0], [20, 1, 0], [20, 2, 0], [21, 1, 0],
                        [21, 2, 0], [23, 1, 0], [23, 2, 0], [24, 1, 0], [24, 2, 0]])
    BC_Disp[:, 0] -= 1  # convert to python indices

    # assemble load vector
    ndn = 2  # 2 DOFs per node
    NDof = ndn * coord.shape[0]  # number of DOFs
    F = np.zeros(NDof)  # initializing right-hand side of equation [K]{u} = {F}
    edge = utils.matlabToPythonIndices(np.array([[32, 33], [33, 34], [34, 36], [36, 37]]))  # edges subject to traction
    trac = np.array([[0, 1E6, 0, 1E6]]).T  # all edges have the same traction (in Pa),
    F = sbfem.addSurfaceTraction(coord, edge, trac, F)

    # solution of S-elements and assemblage of global stiffness and mass matrices
    sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)

    # Static solution of nodal displacements and forces
    d, F = sbfem.solverStatics(K, BC_Disp, F)

    NodalDisp = np.reshape(d, (2, -1), order='F').T
    # Crack opening displacement
    COD = NodalDisp[17-1, :] - NodalDisp[1-1, :]

    return {
        'in': {'coord': coord, 'sdConn': sdConn, 'sdSC': sdSC, 'mat': mat, 'F': F, 'BC_Disp': BC_Disp},
        'out': {'d': d, 'F': F, 'COD': COD}
    }


def example_3_7(isd, xi):
    """
    Example 3.7 A Rectangular Body Under Uniaxial Tension: Internal Displacements and Stresses
    The nodal displacements of the problem shown in Figure 3.3, Example 3.3 have been obtained in Example 3.4.
    Compute the displacements, strains and stresses of S-element 1 at the radial coordinate ξ = 0.5.
    :param isd:  # S-element number
    :param xi:  # radial coordinate
    """
    example = example_3_4()
    mat = example['in']['mat']
    d = example['out']['d']
    sdSln = example['out']['sdSln']

    # strain modes of S-elements
    sdStrnMode = sbfem.strainModesOfSElements(sdSln)

    # integration constants
    sdIntgConst = sbfem.integrationConstsOfSElements(d, sdSln)

    # displacements and strains at specified radial coordinate
    nodexy, dsp, strnNode, GPxy, strnEle = sbfem.displacementsAndStrainsOfSelement(xi, sdSln[isd],
                                                                                   sdStrnMode[isd],
                                                                                   sdIntgConst[isd])

    out1 = example['out']
    out2 = {'sdStrnMode': sdStrnMode, 'sdIntgConst': sdIntgConst,
            'nodexy': nodexy, 'dsp': dsp, 'strnNode': strnNode, 'GPxy': GPxy, 'strnEle': strnEle}
    # merge dicts out1 and out2
    out = {**out1, **out2}
    return {'in': example['in'], 'out': out}

def example_3_8(isd, xi):
    """
    Example 3.8 Patch Test on Distorted Polygon S-elements
    A patch test on a unit square shown in Figure 3.10a is performed.
    :param isd:  # S-element number
    :param xi:  # radial coordinate
    """
    # Mesh
    # nodal coordinates. One node per row [x y]
    x1, y1 = 0.5, 0.5  # Figure b
    # x1, y1 = 0.05, 0.95  # Figure c
    coord = np.array([[x1, y1], [0, 0], [0.1, 0], [1, 0], [1, 1], [0, 1], [0, 0.1]])
    # Input S-element connectivity as a cell array (One S-element per cell).
    # In a cell, the connectivity of line elements is given by one element per row
    # [Node-1 Node-2].
    sdConn = [
        utils.matlabToPythonIndices(np.array([[1, 7], [7, 2], [2, 3], [3, 1]])),  # S-element 1
        utils.matlabToPythonIndices(np.array([[1, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1]]))    # S-element 2
    ]

    # coordinates of scaling centres of S-elements.
    if x1 > y1:  # extension of line 21 intersecting right edge of the square
        sdSC = np.array([[x1/2, y1/2], [(1+x1)/2, y1*(1+x1)/(2*x1)]])
    else:  # extension of line 21 intersecting top edge of the square
        sdSC = np.array([[x1/2, y1/2], [x1*(1+y1)/(2*y1), (1+y1)/2]])

    # Materials
    mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(1, 0.25), den=2)

    # Boundary conditions
    # displacement constrains. One constrain per row: [Node Dir Disp]
    BC_Disp = np.array([
        [utils.matlabToPythonIndices(2), 1, 0],
        [utils.matlabToPythonIndices(2), 2, 0],
        [utils.matlabToPythonIndices(4), 2, 0]
    ])
    # assemble load vector
    ndn = 2  # 2 DOFs per node
    NDof = ndn * len(coord)  # number of DOFs
    F = np.zeros(NDof)  # initializing right-hand side of equation [K]{u} = {F}

    # horizontal tension
    # edge = [ 4 5; 6 7; 7 2];
    # trac = [1 0 1 0; -1 0 -1 0; -1 0 -1 0]’;
    # vertical tension
    # edge = [2 3; 3 4; 5 6];
    # trac = [0 -1 0 -1; 0 -1 0 -1; 0 1 0 1]’;
    # pure shear
    # edges subject to tractions, one row per edge
    edge = utils.matlabToPythonIndices(np.array([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 2]]))
    # tractions, one column per edge
    trac = np.array([[-1, 0, -1, 0],
                     [-1, 0, -1, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0],
                     [0, -1, 0, -1],
                     [0, -1, 0, -1]]).T
    F = sbfem.addSurfaceTraction(coord, edge, trac, F)

    # % Plot mesh
    # plotting options
    # opt = {'LineSpec': '-k', 'sdSC': sdSC, 'PlotNode': 1, 'LabelNode': 1, 'title': 'MESH', 'show': True}
    # utils.plotSBFEMesh(coord, sdConn, opt)

    # Static solution
    # solution of S-elements and assemblage of global stiffness and mass matrices
    sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)
    # Static solution of nodal displacements and forces
    d, F = sbfem.solverStatics(K, BC_Disp, F)
    # nodal displacements
    nodalDisp = np.reshape(d, (-1, 2))

    print('Nodal displacements')
    print(np.hstack((np.expand_dims(np.arange(len(coord), dtype=np.int), axis=1), nodalDisp)))

    # Stresses
    # strain modes of S-elements
    sdStrnMode = sbfem.strainModesOfSElements(sdSln)
    # integration constants
    sdIntgConst = sbfem.integrationConstsOfSElements(d, sdSln)
    # displacements and strains at specified radial coordinate
    nodexy, dsp, strnNode, GPxy, strnEle = sbfem.displacementsAndStrainsOfSelement(xi, sdSln[isd], sdStrnMode[isd], sdIntgConst[isd])

    print('Stresses of Elements 1 and 2')
    stresses = np.matmul(mat.D, strnEle)
    print(stresses[:, :2])

    return {'in': {'coord': coord, 'sdConn': sdConn, 'sdSC': sdSC, 'mat': mat, 'F': F, 'BC_Disp': BC_Disp},
            'out': {
                'nodalDisp': nodalDisp,
                'sdStrnMode': sdStrnMode, 'sdIntgConst': sdIntgConst,
                'nodexy': nodexy, 'dsp': dsp, 'strnNode': strnNode,
                'GPxy': GPxy, 'strnEle': strnEle,
                'stresses': stresses}
            }

