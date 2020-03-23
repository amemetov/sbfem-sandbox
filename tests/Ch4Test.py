import numpy as np
import numpy.testing as npt
import unittest
import examples.Ex4 as Ex4
import sbfem.mesh2d as mesh2d
import sbfem.utils as utils


class Ch4Test(unittest.TestCase):
    """
    This class contains tests for examples and methods from Chapter4.
    """

    def testTriToSBFEMesh(self):
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

        # coord, sdConn, sdSC = mesh2d.

        utils.plotTriFEMesh(p, t, {'LabelEle': 10, 'LabelNode': 10,
                                   'show': True})


    def testMeshConnectivity1(self):
        # triangular elements
        t = utils.matlabToPythonIndices(np.array([[5, 4, 6], [3, 2, 1], [1, 4, 3], [4, 5, 3], [2, 3, 7], [7, 3, 5]]))

        meshEdge, sdEdge = mesh2d.meshConnectivity(t)

        exp_meshEdge = utils.matlabToPythonIndices(
                        np.array([[1, 2], [1, 3], [1, 4],
                                 [2, 3], [2, 7],
                                 [3, 4], [3, 5], [3, 7],
                                 [4, 5], [4, 6],
                                 [5, 6], [5, 7]]))

        exp_sdEdge = utils.matlabToPythonIndices(
                        np.array([[1, 10, 11],
                                  [4,  1,  2],
                                  [3,  6,  2],
                                  [9,  7,  6],
                                  [4,  8,  5],
                                  [8,  7, 12]]))

        npt.assert_equal(meshEdge, exp_meshEdge, err_msg='Mismatched `meshEdge`')
        npt.assert_equal(np.abs(sdEdge), exp_sdEdge, err_msg='Mismatched `sdEdge`')



