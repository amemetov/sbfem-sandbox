import numpy as np
import numpy.testing as npt
import unittest
import sbfem.sbfem as sbfem


class Ch2Test(unittest.TestCase):
    """
    This class contains tests for examples from Chapter2.
    """

    def test_example_2_4(self):
        """
        A square S-element is shown in Figure 2.20. Each edge is modelled by 1 line element.
        The dimensions, nodal numbers and the line element numbers (in circle) are shown in Figure 2.20.
        The elasticity constants are: Young’s modulus E and Poisson’s ratio ν = 0.
        Considering the plane stress states, determine the element coefficient matrices
        [E0e], [E1e] and [E2e] for Element 2 using the equations derived for 2-node line element in Example 2.2.
        """
        xy = np.array([[1, -1], [1, 1]])
        E = 12
        D = E / 2 * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        mat = sbfem.Material(D, 1)
        e0, e1, e2, m0 = sbfem.coeffMatricesOf2NodeLineElement(xy, mat)

        npt.assert_array_equal(e0, np.array([[8., 0., 4., 0.],
                                             [0., 4., 0., 2.],
                                             [4., 0., 8., 0.],
                                             [0., 2., 0., 4.]]))

        npt.assert_array_equal(e1, np.array([[-2., -3., 2., -3.],
                                             [-0., -1., 0., 1.],
                                             [2., 3., -2., 3.],
                                             [0., 1., 0., -1.]]))

        npt.assert_array_equal(e2, np.array([[5., 0., -5., -0.],
                                             [0., 7., -0., -7.],
                                             [-5., -0., 5., 0.],
                                             [-0., -7., 0., 7.]]))

        npt.assert_array_almost_equal(m0, np.array([[0.66666667, 0., 0.33333333, 0.],
                                                    [0., 0.66666667, 0., 0.33333333],
                                                    [0.33333333, 0., 0.66666667, 0.],
                                                    [0., 0.33333333, 0., 0.66666667]]))
