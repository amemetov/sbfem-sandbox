import math
import numpy as np
import sbfem.sbfem as sbfem
import sbfem.mesh2d as mesh2d
import sbfem.utils as utils


class Mesh:
    def __init__(self, coord, vertices):
        # Nodal coordinates. One node per row [x y]
        self.coord = coord

        # Either Triangular elements: One triangle per row [Node-1 Node-2 Node-3] indices of 3 nodes/vertices
        # Or in general Polygon elements: One polygon per row [Node-1, Node-2, ..., Node-N] indices of N nodes/vertices
        self.vertices = vertices


class SBFEMesh:
    def __init__(self, coord, sdConn, sdSC):
        # Nodal coordinates. One node per row [x y]
        self.coord = coord

        # Input S-element connectivity as a cell array/list (One S-element per cell/item).
        # In a cell/item, the connectivity of line elements is given by one element per row [Node-1 Node-2].
        self.sdConn = sdConn

        # Coordinates of scaling centres of S-elements. one S-element per row
        self.sdSC = sdSC


class Ex4Mesh:
    def __init__(self):
        self.inMesh = None
        self.SBFEMesh = None


def example_4_polygonToSBFEMesh():
    """
    p. 160
    """
    # Nodal coordinates
    coord = np.array([[1.00, 1.00], [0.57, 1.00], [0.00, 1.00],
                      [0.53, 0.75], [0.82, 0.50], [1.00, 0.50],
                      [1.00, 0.00], [0.36, 0.63], [0.00, 0.77],
                      [0.53, 0.21], [0.33, 0.38], [0.55, 0.00],
                      [0.00, 0.28], [0.00, 0.00]])

    # see p.157 and p.160
    polygon = utils.matlabToPythonIndices([[9, 8, 4, 2, 3],         # polygon 1
                                           [11, 10, 5, 4, 8],       # polygon 2
                                           [10, 12, 7, 6, 5],       # polygon 3
                                           [14, 12, 10, 11, 13],    # polygon 4
                                           [4, 5, 6, 1, 2],         # polygon 5
                                           [13, 11, 8, 9]])         # polygon 6

    polyMesh = Mesh(coord, polygon)
    sdConn, sdSC = mesh2d.polygonToSBFEMesh(coord, polygon)
    sbfemMesh = SBFEMesh(coord, sdConn, sdSC)

    return {'in': {'polyMesh': polyMesh, 'sbfemMesh': sbfemMesh},
            'out': {}
            }


def example_4_DeepBeam():
    """
    A Deep Beam. (p.178)
    A deep beam with length L = 2 m and height H = 1 m is shown in Figure 4.12.
    The Young’s modulus is E = 10 × 10 6 kPa and the Poisson’s ratio is 0.2.
    The mass density is 2 Mg∕m 3 .
    """

    def TriMesh():
        # nodal coordinates
        p = np.array([[0.00, -0.16], [0.00, 0.16], [0.00, -0.50], [0.00, 0.50],
                      [0.41, -0.50], [0.41, 0.50], [0.51, -0.00], [0.80, -0.50],
                      [0.80, 0.50], [1.00, 0.00], [1.20, 0.50], [1.20, -0.50],
                      [1.49, 0.00], [1.59, 0.50], [1.59, -0.50], [2.00, -0.50],
                      [2.00, 0.50], [2.00, 0.16], [2.00, -0.16]])

        # triangular elements
        t = utils.matlabToPythonIndices(
            np.array([[5, 1, 3], [15, 16, 19], [7, 2, 1], [1, 5, 7],
                      [5, 8, 7], [18, 17, 14], [4, 2, 6], [2, 7, 6],
                      [10, 8, 12], [10, 7, 8], [12, 15, 13], [13, 10, 12],
                      [13, 15, 19], [19, 18, 13], [13, 18, 14], [9, 6, 7],
                      [7, 10, 9], [11, 9, 10], [11, 13, 14], [10, 13, 11]]))
        return p, t

    p, t = TriMesh()
    triMesh = Mesh(p, t)

    coord, sdConn, sdSC = mesh2d.triToSBFEMesh(p, t)
    sbfemMesh = SBFEMesh(coord, sdConn, sdSC)

    return {'in': {'triMesh': triMesh, 'sbfemMesh': sbfemMesh},
            'out': {}
            }


def example_4_L_shaped_Panel():
    """
    An L-shaped Panel. (p.197)
    """

    def TriMesh():
        # nodal coordinates
        p = np.array([
            [-0.0000, 0.4854], [-0.0000, 1.5174], [-0.0000, 1.0308], [0.0000, 0.0000], [0.0000, 2.0000],
            [0.4828, 2.0000], [0.4960, 0.7059], [0.4980, 0.0000], [0.5994, 1.3949], [0.9679, 2.0000],
            [1.0000, 0.4734], [1.0000, 0.0000], [1.0000, 1.0000], [1.2864, 1.5042], [1.5142, 2.0000],
            [1.5267, 1.0000], [2.0000, 1.0000], [2.0000, 2.0000], [2.0000, 1.5021]
        ])

        # triangles
        t = utils.matlabToPythonIndices(
            np.array([
                [19, 14, 16], [16, 14, 13], [13, 7, 11], [19, 18, 15], [15, 14, 19],
                [13, 14, 9], [2, 3, 9], [9, 7, 13], [9, 3, 7], [7, 3, 1],
                [19, 16, 17], [8, 11, 7], [8, 1, 4], [7, 1, 8], [14, 15, 10],
                [10, 9, 14], [11, 8, 12], [9, 10, 6], [6, 5, 2], [2, 9, 6]
            ]))
        return p, t

    p, t = TriMesh()
    triMesh = Mesh(p, t)

    coord, sdConn, sdSC = mesh2d.triToSBFEMesh(p, t)
    sbfemMesh = SBFEMesh(coord, sdConn, sdSC)

    return {'in': {'triMesh': triMesh, 'sbfemMesh': sbfemMesh},
            'out': {}
            }
