import numpy as np
import numpy.testing as npt
import unittest
import sbfem.sbfem as sbfem


class Ch2Test(unittest.TestCase):
    """
    This class contains tests for examples from Chapter2.
    """

    def test_example_2_3(self):
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

    def test_example_2_4(self):
        """
        Example 2.4 Use the function EleCoeff2NodeEle in Code List 2.1 on page 64
        to compute the element coefficient matrices [E0], [E1] and [E2] of the four 2-node line
        elements in the square S-element in Example 2.3 on page 61.
        Assume the Young’s modulus is equal to E = 10 GPa.
        """
        elements = np.array([
            [[-1, -1], [1, -1]],  # Element 1
            [[1, -1],  [1, 1]],   # Element 2
            [[1, 1],   [-1, 1]],  # Element 3
            [[-1, 1],  [-1, -1]]  # Element 4
        ])

        mat = sbfem.Material(D=sbfem.elasticityMatrixForPlaneStress(10, 0), # E in GPa
                       den=2 # mass density in Mg per cubic meter
                      )

        expected_values = np.array(
            [
                # elem 1
                [
                    # e0
                    [
                        [3.33333333, 0., 1.66666667, 0.],
                        [0., 6.66666667, 0., 3.33333333],
                        [1.66666667, 0., 3.33333333, 0.],
                        [0., 3.33333333, 0., 6.66666667]
                    ],
                    # e1
                    [
                        [-0.83333333, 0., 0.83333333, 0.],
                        [2.5, -1.66666667, 2.5, 1.66666667],
                        [0.83333333, 0., -0.83333333, 0.],
                        [-2.5, 1.66666667, -2.5, -1.66666667]
                    ],
                    # e2
                    [
                        [5.83333333, 0., -5.83333333, 0.],
                        [0., 4.16666667, 0., -4.16666667],
                        [-5.83333333, 0., 5.83333333, 0.],
                        [0., -4.16666667, 0., 4.16666667]
                    ]
                ],

                # elem 2
                [
                    # e0
                    [
                        [6.66666667, 0., 3.33333333, 0.],
                        [0., 3.33333333, 0., 1.66666667],
                        [3.33333333, 0., 6.66666667, 0.],
                        [0., 1.66666667, 0., 3.33333333]
                    ],
                    # e1
                    [
                        [-1.66666667, -2.5, 1.66666667, -2.5],
                        [0., -0.83333333, 0., 0.83333333],
                        [1.66666667, 2.5, -1.66666667, 2.5],
                        [0., 0.83333333, 0., -0.83333333]
                    ],
                    # e2
                    [
                        [4.16666667, 0., -4.16666667, 0.],
                        [0., 5.83333333, 0., -5.83333333],
                        [-4.16666667, 0., 4.16666667, 0.],
                        [0., -5.83333333, 0., 5.83333333]
                    ]
                ],

                # elem 3
                [
                    # e0
                    [
                        [3.33333333, 0., 1.66666667, 0.],
                        [0., 6.66666667, 0., 3.33333333],
                        [1.66666667, 0., 3.33333333, 0.],
                        [0., 3.33333333, 0., 6.66666667]
                    ],
                    # e1
                    [
                        [-0.83333333, 0., 0.83333333, 0.],
                        [2.5, -1.66666667, 2.5, 1.66666667],
                        [0.83333333, 0., -0.83333333, 0.],
                        [-2.5, 1.66666667, -2.5, -1.66666667]
                    ],
                    # e2
                    [
                        [5.83333333, 0., -5.83333333, -0.],
                        [0., 4.16666667, 0., -4.16666667],
                        [-5.83333333, 0., 5.83333333, 0.],
                        [0., -4.16666667, 0., 4.16666667]
                    ]
                ],

                # elem 4
                [
                    # e0
                    [
                        [6.66666667, 0., 3.33333333, 0.],
                        [0., 3.33333333, 0., 1.66666667],
                        [3.33333333, 0., 6.66666667, 0.],
                        [0., 1.66666667, 0., 3.33333333]
                    ],
                    # e1
                    [
                        [-1.66666667, -2.5, 1.66666667, -2.5],
                        [0., -0.83333333, 0., 0.83333333],
                        [1.66666667, 2.5, -1.66666667, 2.5],
                        [0., 0.83333333, 0., -0.83333333]
                    ],
                    # e2
                    [
                        [4.16666667, 0., -4.16666667, -0.],
                        [0., 5.83333333, 0., -5.83333333],
                        [-4.16666667, 0., 4.16666667, 0.],
                        [0., -5.83333333, 0., 5.83333333]
                    ]
                ]
            ]
        )

        for xy, exp_val in zip(elements, expected_values):
            e0, e1, e2, m0 = sbfem.coeffMatricesOf2NodeLineElement(xy, mat)
            npt.assert_array_almost_equal(e0, exp_val[0], err_msg=f'Mismatched e0 for {xy}')
            npt.assert_array_almost_equal(e1, exp_val[1], err_msg=f'Mismatched e1 for {xy}')
            npt.assert_array_almost_equal(e2, exp_val[2], err_msg=f'Mismatched e2 for {xy}')

