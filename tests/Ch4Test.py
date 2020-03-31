import numpy as np
import numpy.testing as npt
import unittest
import sbfem.mesh2d as mesh2d
import sbfem.utils as utils


class Ch4Test(unittest.TestCase):
    """
    This class contains tests for examples and methods from Chapter4.
    """

    def testMeshConnectivityTriangularElements(self):
        # see pages 157, 158 and 161
        # triangular elements
        t = utils.matlabToPythonIndices(np.array([[5, 4, 6], [3, 2, 1], [1, 4, 3], [4, 5, 3], [2, 3, 7], [7, 3, 5]]))

        meshEdge, sdEdge, edge2sd, node2Edge, node2sd = mesh2d.meshConnectivity(t)

        exp_meshEdge = utils.matlabToPythonIndices(
            np.array([[1, 2], [1, 3], [1, 4],
                      [2, 3], [2, 7],
                      [3, 4], [3, 5], [3, 7],
                      [4, 5], [4, 6],
                      [5, 6], [5, 7]]))

        exp_sdEdge = utils.matlabToPythonIndices(
            np.array([[9, 10, 11],  # triangle 1
                      [4, 1, 2],    # triangle 2
                      [3, 6, 2],    # triangle 3
                      [9, 7, 6],    # triangle 4
                      [4, 8, 5],    # triangle 5
                      [8, 7, 12]    # triangle 6
                      ]))

        exp_edge2sd = utils.matlabToPythonIndices(
                        [[2],       # edge 1 belongs to S-element 1
                         [2, 3],    # edge 2 belongs to S-elements 2 and 3
                         [3],       # edge 3 belongs to S-element 3
                         [2, 5],    # edge 4 belongs to S-elements 2 and 5
                         [5],       # edge 5 belongs to S-element 5
                         [3, 4],    # edge 6 belongs to S-elements 3 and 4
                         [4, 6],    # edge 7 belongs to S-elements 4 and 6
                         [5, 6],    # edge 8 belongs to S-elements 5 and 6
                         [1, 4],    # edge 9 belongs to S-elements 1 and 4
                         [1],       # edge 10 belongs to S-element 1
                         [1],       # edge 11 belongs to S-element 1
                         [6]        # edge 12 belongs to S-element 6
                         ])

        exp_node2Edge = utils.matlabToPythonIndices([
            [1, 2, 3],          # node 1 belongs to edges 1, 2, 3
            [1, 4, 5],          # node 2 belongs to edges 1, 4, 5
            [2, 4, 6, 7, 8],    # node 3 belongs to edges 2, 4, 5, 7, 8
            [3, 6, 9, 10],      # node 4 belongs to edges 3, 6, 9, 10
            [7, 9, 11, 12],     # node 5 belongs to edges 7, 9, 11, 12
            [10, 11],           # node 6 belongs to edges 10, 11
            [5, 8, 12]          # node 7 belongs to edges 5, 8, 12
        ])

        exp_node2sd = utils.matlabToPythonIndices([
            [2, 3],             # node 1 belongs to S-elements 2, 3
            [2, 5],             # node 2 belongs to S-elements 2, 5
            [2, 3, 4, 5, 6],    # node 3 belongs to S-elements 2, 3, 4, 5, 6,
            [1, 3, 4],          # node 4 belongs to S-elements 1, 3, 4
            [1, 4, 6],          # node 5 belongs to S-elements 1, 4, 6
            [1],                # node 6 belongs to S-elements 1
            [5, 6]              # node 7 belongs to S-elements 5, 6
        ])

        npt.assert_equal(meshEdge, exp_meshEdge, err_msg='Mismatched `meshEdge`')
        npt.assert_equal(np.abs(sdEdge), exp_sdEdge, err_msg='Mismatched `sdEdge`')
        npt.assert_equal(edge2sd, exp_edge2sd, err_msg='Mismatched `edge2sd`')
        npt.assert_equal(node2Edge, exp_node2Edge, err_msg='Mismatched `node2Edge`')
        npt.assert_equal(node2sd, exp_node2sd, err_msg='Mismatched `node2sd`')

    def testMeshConnectivitySElements(self):
        # see p.90
        # Input S-element connectivity as a cell array (One S-element per cell).
        # In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2].
        sdConn = utils.matlabToPythonIndices(
            np.array([
                np.array([[2, 5], [5, 7], [7, 8], [8, 3], [3, 2]]),  # S-element 1
                np.array([[1, 4], [4, 5], [5, 2], [2, 1]]),          # S-element 2
                np.array([[4, 6], [6, 7], [7, 5], [5, 4]])           # S-element 3
            ]))

        meshEdge, sdEdge, _, _, _ = mesh2d.meshConnectivity(sdConn)
        # convert to list of list, otherwise assertSequenceEqual throws exception
        absSdEdge = [e.tolist() for e in np.abs(sdEdge)]

        exp_meshEdge = utils.matlabToPythonIndices(
            np.array([[1, 2], [1, 4], [2, 3], [2, 5], [3, 8], [4, 5], [4, 6], [5, 7], [6, 7], [7, 8]]))
        exp_sdEdge = utils.matlabToPythonIndices([[4, 8, 10, 5, 3],  # S-element 1
                                                  [2, 6, 4, 1],      # S-element 2
                                                  [7, 9, 8, 6]       # S-element 3
                                                  ])

        npt.assert_equal(meshEdge, exp_meshEdge, err_msg='Mismatched `meshEdge`')
        self.assertSequenceEqual(absSdEdge, exp_sdEdge, msg='Mismatched `sdEdge`')

    def testMeshConnectivityPolygonElements(self):
        # see p.157 and p.160
        polygon = utils.matlabToPythonIndices([[9, 8, 4, 2, 3],         # polygon 1
                                               [11, 10, 5, 4, 8],       # polygon 2
                                               [10, 12, 7, 6, 5],       # polygon 3
                                               [14, 12, 10, 11, 13],    # polygon 4
                                               [4, 5, 6, 1, 2],         # polygon 5
                                               [13, 11, 8, 9]])         # polygon 6

        meshEdge, sdEdge, _, _, _ = mesh2d.meshConnectivity(polygon)
        # convert to list of list, otherwise assertSequenceEqual throws exception
        absSdEdge = [e.tolist() for e in np.abs(sdEdge)]

        exp_meshEdge = utils.matlabToPythonIndices(np.array([[1, 2], [1, 6], [2, 3], [2, 4], [3, 9], [4, 5], [4, 8],
                                 [5, 6], [5, 10], [6, 7], [7, 12], [8, 9], [8, 11], [9, 13],
                                 [10, 11], [10, 12], [11, 13], [12, 14], [13, 14]]))

        exp_sdEdge = utils.matlabToPythonIndices([[12,  7,  4,  3,  5],  # polygon 1
                                                  [15,  9,  6,  7, 13],  # polygon 2
                                                  [16, 11, 10,  8,  9],  # polygon 3
                                                  [18, 16, 15, 17, 19],  # polygon 4
                                                  [6, 8, 2, 1, 4],       # polygon 5
                                                  [17, 13, 12, 14]       # polygon 6
                                                  ])

        npt.assert_equal(meshEdge, exp_meshEdge, err_msg='Mismatched `meshEdge`')
        self.assertSequenceEqual(absSdEdge, exp_sdEdge, msg='Mismatched `sdEdge`')
