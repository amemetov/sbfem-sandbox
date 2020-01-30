import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


class Material(object):
    def __init__(self, D, den):
        # elasticity matrix
        self.D = D
        # mass density in Mg per cubic meter
        self.den = den


def elasticityMatrixForPlaneStress(E, p):
    """
    Original name: ElasMtrx
    Computes elasticity matrix (plane stress).
    :param E: Young's modulus in GPa
    :param p: Poisson's ratio
    :return: elsticity matrix
    """
    return (E / (1 - p ** 2)) * np.array([[1, p, 0], [p, 1, 0], [0, 0, (1 - p) / 2]])


def coeffMatricesOf2NodeLineElement(xy, mat: Material):
    """
    Original name: EleCoeff2NodeEle, p. 64
    Coefficient matrices of a 2-node line element
        Inputs:
            xy(i,:)    - coordinates of node i (orgin at scaling centre).
                         The nodes are numbered locally within each line element
            mat        - material constants
              mat.D    - elasticity matrix
              mat.den  - mass density

        Outputs:
            e0, e1, e2, m0  - element coefficient matrices
    """
    dxy = xy[1, :] - xy[0, :]                       # (2.50a), (2.50b)
    mxy = np.sum(xy, axis=0) / 2                    # (2.51a), (2.51b)
    a = xy[0, 0] * xy[1, 1] - xy[1, 0] * xy[0, 1]   # a=2|J_b| (2.58)
    if a < 1e-10:
        raise ValueError('negative area (EleCoeff2NodeEle)')

    C1 = 0.5 * np.array([[dxy[1], 0], [0, -dxy[0]], [-dxy[0], dxy[1]]])     # (2.114a)
    C2 = np.array([[-mxy[1], 0], [0, mxy[0]], [mxy[0], -mxy[1]]])           # (2.114b)

    Q0 = 1 / a * (np.matmul(np.matmul(C1.T, mat.D), C1))  # (2.118a)
    Q1 = 1 / a * (np.matmul(np.matmul(C2.T, mat.D), C1))  # (2.118b)
    Q2 = 1 / a * (np.matmul(np.matmul(C2.T, mat.D), C2))  # (2.118c)

    # element coefficient matrices
    e0 = 2 / 3 * np.block([[2 * Q0, Q0], [Q0, 2 * Q0]])                                 # (2.119a)
    e1 = -1 / 3 * np.block([[Q0, -Q0], [-Q0, Q0]]) + np.block([[-Q1, -Q1], [Q1, Q1]])   # (2.119b)
    e2 = 1 / 3 * np.block([[Q0, -Q0], [-Q0, Q0]]) + np.block([[Q2, -Q2], [-Q2, Q2]])    # (2.119c)

    # mass coefficent matrix
    m0 = a * mat.den / 6 * np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]])  # (3.112)

    return e0, e1, e2, m0
