import math
import numpy as np
import numpy.testing as npt
import unittest
import sbfem.sbfem as sbfem


class Ch3Test(unittest.TestCase):
    """
    This class contains tests for examples from Chapter3.
    """

    def test_example_3_1(self):
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

        exp_K = np.array([
            [4.8936, 1.2500, -2.3936, -1.2500, -2.6064, -1.2500, 0.1064, 1.2500],
            [1.2500, 4.8936, 1.2500, 0.1064, -1.2500, -2.6064, -1.2500, -2.3936],
            [-2.3936, 1.2500, 4.8936, -1.2500, 0.1064, -1.2500, -2.6064, 1.2500],
            [-1.2500, 0.1064, -1.2500, 4.8936, 1.2500, -2.3936, 1.2500, -2.6064],
            [-2.6064, -1.2500, 0.1064, 1.2500, 4.8936, 1.2500, -2.3936, -1.2500],
            [-1.2500, -2.6064, -1.2500, -2.3936, 1.2500, 4.8936, 1.2500, 0.1064],
            [0.1064, -1.2500, -2.6064, 1.2500, -2.3936, 1.2500, 4.8936, -1.2500],
            [1.2500, -2.3936, 1.2500, -2.6064, -1.2500, 0.1064, -1.2500, 4.8936]
        ])

        exp_d = np.array([1.915, 1.915, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

        # See the p.81 in the book:
        # "The eigenvectors are not uniquely defined.
        # A linear combination of the eigenvectors having the same eigenvalue is also an eigenvector.
        # So the eigenvector matrix may vary in value."
        # That's why MATLAB's result is not equal to numpy's result
        exp_v_matlab = np.array([
            [-1.0000, -0.1512, -0.3740, 0.4303, 1.0000, 0.3174, 1.0000, 0],
            [0.1741, 0.9923, 0.3444, -0.0109, -0.2360, 1.0000, 0, 1.0000],
            [1.0000, -0.1956, -1.0000, 1.0000, 0.1011, 0.3477, 1.0000, 0],
            [0.1741, -1.0000, -0.0325, 0.3071, 0.2360, -0.9180, 0, 1.0000],
            [-1.0000, -0.1512, 0.3740, -0.4303, -1.0000, -0.3174, 1.0000, 0],
            [0.1741, 0.9923, -0.3444, 0.0109, 0.2360, -1.0000, 0, 1.0000],
            [1.0000, -0.1956, 1.0000, -1.0000, -0.1011, -0.3477, 1.0000, 0],
            [0.1741, -1.0000, 0.0325, -0.3071, -0.2360, 0.9180, 0, 1.0000],
        ])

        exp_M = np.array([
            [0.8954, 0.0851, 0.4380, -0.0000, 0.2287, 0.0851, 0.4380, 0.0000],
            [0.0851, 0.8954, -0.0000, 0.4380, 0.0851, 0.2287, -0.0000, 0.4380],
            [0.4380, 0.0000, 0.8954, -0.0851, 0.4380, -0.0000, 0.2287, -0.0851],
            [-0.0000, 0.4380, -0.0851, 0.8954, -0.0000, 0.4380, -0.0851, 0.2287],
            [0.2287, 0.0851, 0.4380, -0.0000, 0.8954, 0.0851, 0.4380, -0.0000],
            [0.0851, 0.2287, 0.0000, 0.4380, 0.0851, 0.8954, -0.0000, 0.4380],
            [0.4380, -0.0000, 0.2287, -0.0851, 0.4380, 0.0000, 0.8954, -0.0851],
            [0.0000, 0.4380, -0.0851, 0.2287, -0.0000, 0.4380, -0.0851, 0.8954]
        ])

        npt.assert_array_almost_equal(K, exp_K, decimal=4, err_msg=f"Mismatched 'K'")
        npt.assert_array_almost_equal(d, exp_d, decimal=4, err_msg=f"Mismatched 'd'")
        # npt.assert_array_almost_equal(v, exp_v_matlab, decimal=4, err_msg=f"Mismatched 'v'")
        npt.assert_array_almost_equal(M, exp_M, decimal=4, err_msg=f"Mismatched 'M'")

    def test_example_3_2(self):
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

        exp_K = np.array([
            [4.8887, 0.9045, -1.9539, -1.3555, -1.8823, -0.7338, -0.8190, -0.7297, -0.2335, 1.9145],
            [0.9045, 5.4765, 1.3555, -0.0518, -1.0752, -1.2945, -0.3883, -2.3578, -0.7965, -1.7723],
            [-1.9539, 1.3555, 4.8887, -0.9045, -0.2335, -1.9145, -0.8190, 0.7297, -1.8823, 0.7338],
            [-1.3555, -0.0518, -0.9045, 5.4765, 0.7965, -1.7723, 0.3883, -2.3578, 1.0752, -1.2945],
            [-1.8823, -1.0752, -0.2335, 0.7965, 5.9520, 0.5590, -1.2968, -0.4510, -2.5395, 0.1707],
            [-0.7338, -1.2945, -1.9145, -1.7723, 0.5590, 4.4132, 2.2600, -0.7090, -0.1707, -0.6374],
            [-0.8190, -0.3883, -0.8190, 0.3883, -1.2968, 2.2600, 4.2316, -0.0000, -1.2968, -2.2600],
            [-0.7297, -2.3578, 0.7297, -2.3578, -0.4510, -0.7090, -0.0000, 6.1337, 0.4510, -0.7090],
            [-0.2335, -0.7965, -1.8823, 1.0752, -2.5395, -0.1707, -1.2968, 0.4510, 5.9520, -0.5590],
            [1.9145, -1.7723, 0.7338, -1.2945, 0.1707, -0.6374, -2.2600, -0.7090, -0.5590, 4.4132]
        ])

        exp_d = np.array([2.1922, 2.1922, 2.1731, 2.1731, 1.0, 1.0, 1.0, 1.0, 0, 0])

        # See the p.81 in the book:
        # "The eigenvectors are not uniquely defined.
        # A linear combination of the eigenvectors having the same eigenvalue is also an eigenvector.
        # So the eigenvector matrix may vary in value."
        # That's why MATLAB's result is not equal to numpy's result
        exp_v_matlab = np.array([
            [1.0000, 0.1839, -0.8822, -0.4810, -0.0868, -0.1261, -1.0000, 0.8781, 1.0000, 0],
            [-0.3169, -0.7495, 0.2330, 0.7294, 0.7494, 0.8352, 0.1016, 0.0585, 0, 1.0000],
            [-0.9650, -0.3317, 0.5767, -0.4590, 0.0154, 0.0446, -0.5582, -0.1972, 1.0000, 0],
            [-0.6136, 0.5053, -0.2410, -0.7295, -0.4055, 0.7829, -0.3488, 0.2849, 0, 1.0000],
            [0.1610, 0.6518, -0.3249, 0.3807, 0.0963, 0.1536, 0.6550, -1.0000, 1.0000, 0],
            [0.9478, -0.4567, 1.0000, 0.4637, -1.0000, -0.3513, -0.3172, 0.1176, 0, 1.0000],
            [0.2365, -1.0000, -0.3249, -1.0000, 0.0441, 0.0504, 0.9630, -0.4208, 1.0000, 0],
            [-0.6508, -0.1708, -0.5340, -0.0080, -0.2125, -1.0000, 0.1528, -0.2122, 0, 1.0000],
            [-0.4325, 0.4960, 0.5767, 0.3943, -0.0691, -0.1225, -0.0598, 0.7399, 1.0000, 0],
            [0.6336, 0.8717, 0.7070, -0.4380, 0.8687, -0.2667, 0.4117, -0.2488, 0, 1.0000]
        ])

        exp_M = np.array([
            [0.3694, 0.0658, 0.1803, 0.0035, 0.1027, 0.0380, 0.0514, 0.0327, 0.2008, 0.0032],
            [0.0658, 0.4121, -0.0035, 0.2030, 0.0493, 0.0743, 0.0213, 0.1257, 0.0102, 0.1824],
            [0.1803, -0.0035, 0.3694, -0.0658, 0.2008, -0.0032, 0.0514, -0.0327, 0.1027, -0.0380],
            [0.0035, 0.2030, -0.0658, 0.4121, -0.0102, 0.1824, -0.0213, 0.1257, -0.0493, 0.0743],
            [0.1027, 0.0493, 0.2008, -0.0102, 0.4467, 0.0406, 0.1881, 0.0143, 0.1344, -0.0057],
            [0.0380, 0.0743, -0.0032, 0.1824, 0.0406, 0.3348, 0.0073, 0.1951, 0.0057, 0.0426],
            [0.0514, 0.0213, 0.0514, -0.0213, 0.1881, 0.0073, 0.3216, -0.0000, 0.1881, -0.0073],
            [0.0327, 0.1257, -0.0327, 0.1257, 0.0143, 0.1951, -0.0000, 0.4599, -0.0143, 0.1951],
            [0.2008, 0.0102, 0.1027, -0.0493, 0.1344, 0.0057, 0.1881, -0.0143, 0.4467, -0.0406],
            [0.0032, 0.1824, -0.0380, 0.0743, -0.0057, 0.0426, -0.0073, 0.1951, -0.0406, 0.3348]
        ])

        npt.assert_array_almost_equal(K, exp_K, decimal=4, err_msg=f"Mismatched 'K'")
        npt.assert_array_almost_equal(d, exp_d, decimal=4, err_msg=f"Mismatched 'd'")
        # npt.assert_array_almost_equal(np.real(v), np.real(exp_v_matlab), decimal=4, err_msg=f"Mismatched 'v'")
        npt.assert_array_almost_equal(M, exp_M, decimal=4, err_msg=f"Mismatched 'M'")


    def test_example_3_3(self):
        """
        Example 3.3 Rectangular Body Under Uniaxial Tension: Assembly of Global Equations and Solution
        A rectangular body is modelled by 3 S-elements as shown in Figure 3.3a.
        The dimensions (Unit: m) are indicated in Figure 3.3 with b = 1.
        The material constant are Young’s modulus E = 10 GPa and Poisson’s ratio ν = 0.25.
        The input of the S-element data and the assembly of the global stiffness matrix are illustrated.
        """
        example3 = self._example3SElements()
        coord = example3['coord']
        sdConn = example3['sdConn']
        sdSC = example3['sdSC']
        mat = example3['mat']

        # solution of S-elements and assemblage of global stiffness and mass matrices
        sdSln, K, M = sbfem.sbfemAssembly(coord, sdConn, sdSC, mat)

        expected_K = {
            (1, 1): 0.4692, (2, 1): 0.1667, (3,1): 0.0642, (4,1): 0.0333,
            (7,1): -0.2692, (8,1): -0.0333, (9,1): -0.2642, (10,1): -0.1667,
            (1,2): 0.1667, (2,2): 0.4692, (3,2): -0.0333, (4,2): -0.2692,
            (7,2): 0.0333, (8,2): 0.0642, (9,2): -0.1667, (10,2): -0.2642,
            (1,3): 0.0642, (2,3): -0.0333, (3,3): 1.0227, (4,3): -0.0688,
            (5,3): 0.1143, (6,3): 0.0009, (7,3): -0.2642, (8,3): 0.1667,
            (9,3): -0.5381, (10,3): 0.0357, (13,3): -0.1846, (14,3): 0.0331, (15,3): -0.2143, (16,3): -0.1343,
            (1,4): 0.0333, (2,4): -0.2692, (3,4): -0.0688, (4,4): 0.8573, (5,4): -0.0988,
            (6,4): -0.1353, (7,4): 0.1667, (8,4): -0.2642, (9,4): 0.1019, (10,4): -0.0416,
            (13,4): -0.0331, (14,4): -0.0157, (15,4): -0.1012, (16,4): -0.1314,
            (3,5): 0.1143, (4,5): -0.0988, (5,5): 0.4691, (6,5): -0.1667, (9,5): -0.1000,
            (10,5): 0.1309, (13,5): -0.2143, (14,5): 0.1012, (15,5): -0.2691, (16,5): 0.0333,
            (3,6): 0.0009, (4,6): -0.1353, (5,6): -0.1667, (6,6): 0.4686, (9,6): 0.0648,
            (10,6): -0.2667, (13,6): 0.1343, (14,6): -0.1314, (15,6): -0.0333, (16,6): 0.0647,
            (1,7): -0.2692, (2,7): 0.0333, (3,7): -0.2642, (4,7): 0.1667, (7,7): 0.9383, (8,7): 0.0000,
            (9,7): 0.1284, (10,7): -0.0000, (11,7): -0.2692, (12,7): -0.0333, (13,7): -0.2642, (14,7): -0.1667,
            (1,8): -0.0333, (2,8): 0.0642, (3,8): 0.1667, (4,8): -0.2642, (7,8): 0.0000, (8,8): 0.9383,
            (9,8): 0.0000, (10,8): -0.5383, (11,8): 0.0333, (12,8): 0.0642, (13,8): -0.1667, (14,8): -0.2642,
            (1,9): -0.2642, (2,9): -0.1667, (3,9): -0.5381, (4,9): 0.1019, (5,9): -0.1000, (6,9): 0.0648,
            (7,9): 0.1284, (8,9): 0.0000, (9,9): 1.6762, (10,9): 0.0000, (11,9): -0.2642, (12,9): 0.1667,
            (13,9): -0.5381, (14,9): -0.1019, (15,9): -0.1000, (16,9): -0.0648,
            (1,10): -0.1667, (2,10): -0.2642, (3,10): 0.0357, (4,10): -0.0416, (5,10): 0.1309,
            (6,10): -0.2667, (7,10): -0.0000, (8,10): -0.5383, (9,10): 0.0000, (10,10): 1.6831,
            (11,10): 0.1667, (12,10): -0.2642, (13,10): -0.0357, (14,10): -0.0416, (15,10): -0.1309, (16,10): -0.2667,
            (7,11): -0.2692, (8,11): 0.0333, (9,11): -0.2642, (10,11): 0.1667, (11,11): 0.4692,
            (12,11): -0.1667, (13,11): 0.0642, (14,11): -0.0333,
            (7,12): -0.0333, (8,12): 0.0642, (9,12): 0.1667, (10,12): -0.2642,
            (11,12): -0.1667, (12,12): 0.4692, (13,12): 0.0333, (14,12): -0.2692,
            (3,13): -0.1846, (4,13): -0.0331, (5,13): -0.2143, (6,13): 0.1343,
            (7,13): -0.2642, (8,13): -0.1667, (9,13): -0.5381, (10,13): -0.0357, (11,13): 0.0642,
            (12,13): 0.0333, (13,13): 1.0227, (14,13): 0.0688, (15,13): 0.1143, (16,13): -0.0009,
            (3,14): 0.0331, (4,14): -0.0157, (5,14): 0.1012, (6,14): -0.1314, (7,14): -0.1667,
            (8,14): -0.2642, (9,14): -0.1019, (10,14): -0.0416, (11,14): -0.0333, (12,14): -0.2692,
            (13,14): 0.0688, (14,14): 0.8573, (15,14): 0.0988, (16,14): -0.1353,
            (3,15): -0.2143, (4,15): -0.1012, (5,15): -0.2691, (6,15): -0.0333, (9,15): -0.1000,
            (10,15): -0.1309, (13,15): 0.1143, (14,15): 0.0988, (15,15): 0.4691, (16,15): 0.1667,
            (3,16): -0.1343, (4,16): -0.1314, (5,16): 0.0333, (6,16): 0.0647, (9,16): -0.0648,
            (10,16): -0.2667, (13,16): -0.0009, (14,16): -0.1353, (15,16): 0.1667, (16,16): 0.4686,
        }

        for (matlabStartIdx, matlabEndIdx), expectedValue in expected_K.items():
            s, e = matlabStartIdx - 1, matlabEndIdx - 1
            actualVal = K[s, e]
            self.assertAlmostEqual(1.0e-07 * actualVal, expectedValue, places=4, msg=f"s: {matlabStartIdx}, e: {matlabEndIdx}")

    def _example3SElements(self):
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
                'BC_Frc': BC_Frc,
                'F': F,
                'BC_Disp': BC_Disp
        }
