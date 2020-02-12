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
    :return: elasticity matrix
    """
    # E/(1-pˆ2)*[1 p 0; p 1 0; 0 0 (1-p)/2];
    return (E / (1 - p ** 2)) * np.array([[1, p, 0], [p, 1, 0], [0, 0, (1 - p) / 2]])


def coeffMatricesOf2NodeLineElement(xy, mat: Material):
    """
    Original name: EleCoeff2NodeEle (p.64)
    Coefficient matrices of a 2-node line element
    :param xy[i, :]: coordinates of node i (origin at scaling centre).
                    The nodes are numbered locally within each line element
    :param mat: material constants (elasticity matrix, mass density)
    :return: e0, e1, e2, m0  - element coefficient matrices
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

def coeffMatricesOfSElement(xy, conn, mat: Material):
    """
    Original name: SElementCoeffMtx (p.67)
    :param xy[i, :]: coordinates of node i (origin at scaling centre)
                    The nodes are numbered locally within an S-element starting from 0
    :param conn[ie, :]: local connectivity matrix of line element ie in the local nodal numbers of an S-element
    :param mat: material constants (elasticity matrix, mass density)
    :return: E0, E1, E2, M0 - coefficient matrices of S-element
    """

    # number of DOFs at boundary (2 DOFs per node)
    nd = 2 * xy.shape[0]

    # initialising variables
    E0 = np.zeros((nd, nd), dtype=np.float32)
    E1 = np.zeros((nd, nd), dtype=np.float32)
    E2 = np.zeros((nd, nd), dtype=np.float32)
    M0 = np.zeros((nd, nd), dtype=np.float32)

    # loop over elements at boundary
    for ie, elem_nodes in enumerate(conn):
        # nodal coordinates of an element
        xyEle = xy[elem_nodes]

        # get element coefficient matrices of an element
        ee0, ee1, ee2, em0 = coeffMatricesOf2NodeLineElement(xyEle, mat)

        # local DOFs (in S-element) of an element
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        d = np.array([[2*nodeIdx, 2*nodeIdx + 1] for nodeIdx in elem_nodes]).flatten()

        # assemble coefficient matrices of S-element
        E0[d[:, np.newaxis], d] += ee0
        E1[d[:, np.newaxis], d] += ee1
        E2[d[:, np.newaxis], d] += ee2
        M0[d[:, np.newaxis], d] += em0

    return E0, E1, E2, M0


def sbfem(E0, E1, E2, M0):
    """
    Original name: SElementSlnEigenMethod (p.79)
    :param E0: coefficient matrix of S-Element
    :param E1: coefficient matrix of S-Element
    :param E2: coefficient matrix of S-Element
    :param M0: coefficient matrix of S-Element
    :return: a tuple (K, d, v11, M) where
    K   - stiffness matrix of S-Element
    d   - eigenvalues
    v11 - upper half of eigenvectors (displacement modes)
    M   - mass matrix of S-Element
    """

    # number of DOFs of boundary nodes
    nd = E0.shape[0]

    # Preconditioning
    # Eq.(3.28)
    Pf = 1. / np.sqrt(np.abs(np.diag(E0)))
    P = np.diag(Pf)
    # Eq.(3.27)
    E0 = P.dot(E0).dot(P)
    E1 = P.dot(E1).dot(P)
    E2 = P.dot(E2).dot(P)

    # Construct Zp - Eq.(3.30)
    E0_inv = np.linalg.inv(E0)
    m1 = np.concatenate((-E0_inv.dot(E1.T), E0_inv), axis=1)
    m2 = np.concatenate((E2 + E1.dot(m1[:, :nd]), -m1[:, :nd].T), axis=1)
    Zp = np.concatenate((m1, m2))

    # eigenvalues and eigenvectors - Eq.(3.29)
    d, v = np.linalg.eig(Zp)

    # index for sorting eignvalues in descending order of real part.
    idx = np.argsort(np.real(d))[::-1]

    # select eigenvalues and eigenvectors for solution of bounded domain
    # select the first half of sorted eigenvalues
    d = d[idx[:nd]]
    # select the corresponding eigenvectors
    v = v[:, idx[:nd]]
    v = np.diag(np.concatenate((Pf, 1./Pf), axis=0)).dot(v) # Eq.(3.5)

    # modes of translational rigid body motion, see Item 2 on page 75
    # set last two eigenvalues to zero
    d[-2:] = 0
    # set last two eigenvectors to zero
    v[:, -2:] = 0
    # set u_{x}=1 in {phi^(u)_{n-1}}
    v[0:nd:2, -2] = 1
    # set u_{y}=1 in {phi^(u)_{n}}
    v[1:nd:2, -1] = 1

    # normalization of eigenvectors - Eq.(3.37)
    v = v / np.max(np.abs(v[0:nd-2, :]), axis=0)
    v11 = v[0:nd, :]
    v11inv = np.linalg.inv(v11)

    # stiffness matrix - Eq. (3.25)
    K = np.real(v[nd:, :].dot(v11inv))

    # mass matrix
    M0 = v11.T.dot(M0).dot(v11) # Eq. (3.103)
    # am is a square matrix with all columns being the vector of eigenvalues
    # The entry (i, j) equals λ_i.
    # The entry (i, j) of am' equals λ_j
    am = np.tile(np.expand_dims(d, 1), (1, nd))
    # the entry (i, j) of am+am' equals to λ_i + λ_j
    M0 = M0 / (2+am+am.T)
    # Eq. (3.105)
    M = np.real(v11inv.T.dot(M0).dot(v11inv))

    return K, d, v11, M
