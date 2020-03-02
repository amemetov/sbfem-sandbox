import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix
import scipy.sparse.linalg


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

def sbfemAssembly(coord, sdConn, sdSC, mat):
    """
    Original name: SBFEMAssembly (p.88)
    Assembly of global stiffness and mass matrices.
    :param coord: coord[i,:] - coordinates of node i
    :param sdConn: sdConn{isd,:}(ie,:) - S-element connectivity. The nodes of line element ie in S-element isd.
    :param sdSC: sdSC(isd,:) - coordinates of scaling centre of S-element isd
    :param mat: material constants (elasticity matrix, mass density)
    :return: sdSln, K, M
    sdSln - solutions for S-element
    K - global stiffness matrix
    M - global mass matrix
    """

    # Solution of subdomains

    # number of S-elements
    Nsd = len(sdConn)

    # store solutions for S-elements
    sdSln = []

    # loop over S-elements
    for isd in range(Nsd):
        # sdNode contains global nodal numbers of the nodes in an S-element.
        # Vector ic maps the global connectivity to the local connectivity of the S-element
        sdNode, ic = np.unique(sdConn[isd].flatten(), return_inverse=True) # remove duplicates
        xy = coord[sdNode]  # nodal coordinates
        # transform coordinate origin to scaling centre
        xy = xy - sdSC[isd]
        # line element connectivity in local nodal numbers of an S-element
        LConn = np.reshape(ic, sdConn[isd].shape)
        # compute S-element coefficient matrices
        E0, E1, E2, M0 = coeffMatricesOfSElement(xy, LConn, mat)
        # compute solution for S-element
        K, d, v, M = sbfem(E0, E1, E2, M0)
        # store S-element data and solution
        sdSln.append({
            'xy': xy,
            'sc': sdSC[isd],
            'conn': LConn,
            'node': sdNode,
            'K': K, 'M': M, 'd': d, 'v': v
        })

    # Assembly
    # sum of entries of stiffness matrices of all S-elements
    ncoe = 0
    for sln in sdSln:
        ncoe += sln['K'].size

    # initializing non-zero entries in global stiffness and mass matrix
    K = np.zeros(ncoe)
    M = np.zeros(ncoe)
    # rows and columns of non-zero entries in global stiffness matrix
    Ki = np.zeros(ncoe, dtype=np.int32)
    Kj = np.zeros(ncoe, dtype=np.int32)

    StartInd = 0  # starting position of an S-element stiffness matrix
    # loop over subdomains
    for sln in sdSln:
        # global DOFs of nodes in an S-element
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        dof = np.concatenate((np.reshape(2 * sln['node'], (-1, 1)),
                              np.reshape(2 * sln['node'] + 1, (-1, 1))), axis=1).reshape((-1, 1))
        # number of DOFs of an S-element
        Ndof = dof.shape[0]
        # row and column numbers of stiffness coefficients of an S-element
        sdI = np.tile(dof, (1, Ndof))
        sdJ = sdI.T

        # store stiffness, row and column indices
        EndInd = StartInd + Ndof**2 # ending position
        K[StartInd:EndInd] = sln['K'].flatten()
        M[StartInd:EndInd] = sln['M'].flatten()
        Ki[StartInd:EndInd] = sdI.flatten()
        Kj[StartInd:EndInd] = sdJ.flatten()

        StartInd = EndInd  # increment the starting position

    # form global stiffness matrix in sparse storage
    K = sparse_matrix((K, (Ki, Kj)))
    # ensure symmetry
    K = (K+K.T)/2
    # form global mass matrix in sparse storage
    M = sparse_matrix((M, (Ki,Kj)))
    # ensure symmetry
    M = (M+M.T)/2

    return sdSln, K, M

def addNodalForces(BC_Frc, F):
    """
    Original name: AddNodalForces (p.95)
    Assembly of prescribed nodal forces to load vector.
    :param BC_Frc: BC_Frc(i,:) - one force component per row [Node Dir F],
    where Dir can be 1 for x-direction and 2 for y-direction
    :param F: nodal force vector
    :return: nodal force vector
    """
    # 2 DOFs per node
    ndn = 2
    if len(BC_Frc) > 0:
        # DOFs
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        node = BC_Frc[:, 0]
        prevNode = node - 1
        prevNodeUy = ndn*prevNode + 1
        fdof = prevNodeUy + BC_Frc[:, 1]
        fdof = fdof.astype(np.int)

        # accumulate forces
        F[fdof] = F[fdof] + BC_Frc[:, 2]

    return F


def addSurfaceTraction(coord, edge, trac, F):
    """
    Original name: addSurfTraction (p.95)
    Assembly of surface tractions as equivalent nodal forces to load vector
    :param coord: coord(i,:) - coordinates of node i
    :param edge: edge(i,1:2) - the 2 nodes of edge i
    :param trac: trac - surface traction
                when trac has only one column trac(1:4) - surface traction at the 2 nodes of all edges
                when trac has more than one column trac(1:4,i) - surface traction at the 2 nodes of edge i
    :param F: global load vector
    :return: global load vector
    """

    if trac.shape[1] == 1:  # expand uniform surface traction to all edges
        np.tile(trac, (1, edge.shape[0]))

    # equivalent nodal forces
    fmtx = np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]])  # see Eq.3.49
    edgeLen = np.sqrt(np.sum((coord[edge[:, 1], :] -  coord[edge[:,0], :])**2, 1))  # edge length
    nodalF = 1/6*np.matmul(fmtx, trac)*np.tile(np.expand_dims(edgeLen, 1), (1, 4)).T  # Eq. (3.49)

    # assembly of nodal forces
    for ii in range(edge.shape[0]):
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        dofs = np.array([2 * edge[ii, 0], 2 * edge[ii, 0] + 1,
                         2 * edge[ii, 1], 2 * edge[ii, 1] + 1])
        F[dofs] += nodalF[:, ii]

    return F

def solverStatics(K, BC_Disp, F):
    """
    Original name: SolverStatics (p.97)
    2D Static Analysis by the Scaled Boundary Finite Element Method.
    :param K: static stiffness matrix
    :param BC_Disp: prescribed displacements in rows of [Node, Direction (=1 for x; =2 for y), Displacement]
    :param F: external load vector
    :return: (d, F) where
    d - nodal displacements
    F - external nodal forces including support reactions
    """
    ndn = 2 # 2 DOFs per node
    NDof = K.shape[0]
    # Initialization of nodal displacements
    d = np.zeros(NDof)

    # enforcing displacement boundary condition
    # initialization of unconstrained (free) DOFs with unknown displacements
    FDofs = np.arange(NDof)
    if len(BC_Disp) > 0:
        # constrained DOFs with prescribed displacements
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        node = BC_Disp[:, 0]
        prevNode = node - 1
        prevNodeUy = ndn*prevNode + 1
        CDofs = prevNodeUy + BC_Disp[:, 1]
        # remove constrained DOFs
        # FDofs[CDofs] = []
        FDofs = np.delete(FDofs, CDofs)
        F = F - K[:,CDofs]*BC_Disp[:, 2]  # Eq. (3.52)
        # store prescribed displacements
        d[CDofs] = BC_Disp[:, 2]

    # displacement of free DOFs, see Eq. (3.53)
    # the origin code uses matrix left division
    # which is a solution of Ax = B for x
    # for numpy: linalg.solve(a,b) if a is square; linalg.lstsq(a,b) otherwise
    d[FDofs] = scipy.sparse.linalg.spsolve(K[FDofs, :][:, FDofs], F[FDofs])

    # external forces, see Eq. (3.51)
    F = K * d

    return d, F


def strainModesOfSElements(sdSln):
    """
    Original name: SElementStrainMode2NodeEle (p.115)
    Interal displacements and strain modes of S-elements.
    :param sdSln: solutions for S-elements
    :return: strain modes of S-elements
    """
    Nsd = len(sdSln)  # number of S-elements
    sdStrnMode = []  # initialisation of output argument
    for isd in range(Nsd):  # loop over S-elements
        # number of DOFs of boundary nodes
        nd = 2*len(sdSln[isd]['node'])
        v = sdSln[isd]['v']  # displacement modes

        # Strain modes. The last 2 columns equal 0 (rigid body motions)
        # number of DOFs excluding 2 translational rigid-body motions
        nd2 = nd - 2
        d = sdSln[isd]['d'][:nd2]  # eigenvalues

        # See Eq. (3.66). [Φ(u)b]⟨λb⟩ is computed.
        # The element values are extracted based on element connectivity

        vb = v[:, :nd2] * np.tile(np.expand_dims(d[:nd2], 1), (1, nd)).T

        n1 = sdSln[isd]['conn'][:, 0]  # first node of all 2-node elements
        n2 = sdSln[isd]['conn'][:, 1]  # second node of all 2-node elements
        # LDof(i,:): Local DOFs of nodes of element i in an S-element
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        LDof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1]).T

        xy = sdSln[isd]['xy']  # nodal coordinates with origin at scaling centre
        # dxy(i,:):[Δx, Δy] of i-th element, Eq. (2.50)
        dxy = xy[n2, :] - xy[n1, :]
        # mxy(i,:):[x̄, ȳ] of i-th element, Eq. (2.51)
        mxy = (xy[n2, :] + xy[n1, :]) / 2
        # a(i): 2|Jb| of i-th element, Eq. (2.57)
        a = mxy[:, 0] * dxy[:, 1] - mxy[:, 1] * dxy[:, 0]

        ne = len(n1)  # number of line elements

        # initializing strain modes
        # create a matrix containing complex numbers
        # otherwise casting complex values to real discards the imaginary part
        # and that leads to computing error
        mode = np.zeros((3*ne, nd), dtype="complex_")

        for ie in range(ne):  # loop over elements at boundary
            C1 = 0.5 * np.array([[dxy[ie, 1], 0], [0, -dxy[ie, 0]], [-dxy[ie, 0], dxy[ie, 1]]])  # (2.114a)
            C2 = np.array([[-mxy[ie, 1], 0], [0, mxy[ie, 0]], [mxy[ie, 0], -mxy[ie, 1]]])  # (2.114b)
            B1 = 1/a[ie] * np.hstack((C1, C1))  # Eq. (3.74a)
            B2 = 1/a[ie] * np.hstack((-C2, C2))  # Eq. (3.74b)
            mode[3*ie:3*(ie + 1), :nd2] = np.matmul(B1, vb[LDof[ie, :], :]) + np.matmul(B2, v[LDof[ie, :], :nd2])  # strain modes, Eq (3.66)

        # Store the ouput in cell array sdPstP.
        # The number of S-element isd is the index of the array.
        # GPxy(ie,:): the coordinates of the Gauss Point of element ie (middle point of 2-node element).
        # strnMode(:,ie): the strain modes at the Gauss Point of element ie.

        sdStrnMode.append({'xy': mxy, 'value': mode})

    return sdStrnMode


def integrationConstsOfSElements(d, sdSln):
    """
    Original name: SElementIntgConst (p.115)
    Integration constants of S-elements
    :param d: nodal displacements
    :param sdSln: solutions for S-elements
    :return: vector of integration constants
    """
    Nsd = len(sdSln)  # total number of S-elements
    sdIntgConst = []  # initialization of output argument
    for isd in range(Nsd):  # loop over S-elements
        # Integration constants
        # global DOFs of nodes in an S-element
        # for Python        # Original
        # for ux => 2*i     # 2*i -1
        # for uy => 2*i + 1 # 2*i
        dof = np.vstack((2*sdSln[isd]['node'], 2*sdSln[isd]['node'] + 1)).flatten(order='F')

        dsp = d[dof]  # nodal displacements at boundary of S-element

        # the origin code uses matrix left division
        # which is a solution of Ax = B for x
        sdIntgConst.append(np.linalg.solve(sdSln[isd]['v'], dsp))  # integration constants, see Eq. (3.56)

    return sdIntgConst


def displacementsAndStrainsOfSelement(xi, sdSln, sdStrnMode, sdIntgConst):
    """
    Original name: SElementInDispStrain (p.117)
    Displacements and strains at specified radial coordinate
    :param xi: radial coordinate
    :param sdSln: solutions for S-element
    :param sdStrnMode: strain modes of S-element
    :param sdIntgConst: integration constants
    :return: a tuple (nodexy, dsp, strnNode, GPxy, strnEle)
            where (All valus are on the scaled boundary, i.e. coodinate line, at specified xi):
                nodexy(i,:)   - coordinates of node i
                dsp(i,:)      - nodal displacement funcitons of node i
                strnNode(:,i) - strains on the radial line of node i
                GPxy(ie,:)    - coordinates of middle point of element ie
                strnEle(:,ie) - strains at middle point of element ie
    """

    # Transform local coordinates (origin at scaling centre) at scaled boundary to global coordinates.
    # GPxy(ie,:) - coordinates of Gauss point (middle of 2-node element) of element ie after scaling.
    # nodexy(i,:) - coordinates of node i after scaling.

    GPxy = xi * sdStrnMode['xy'] + sdSln['sc']
    nodexy = xi * sdSln['xy'] + sdSln['sc']

    if xi > 1.E-16:  # outside of a tiny region around the scaling centre
        fxi = (xi**sdSln['d']) * sdIntgConst  # ξ⟨λb⟩{c}
        dsp = np.matmul(sdSln['v'], fxi) # Eq.(3.21a)
        strnEle = np.matmul(sdStrnMode['value'][:, :-2], fxi[:-2]) / xi  # Eq.(3.67)
    else:  # at scaling centre
        dsp = np.matmul(sdSln['v'][:, -1:], sdIntgConst[-1:])  # Eq. (3.57)
        if(np.min(np.real(sdSln['d'][0: -2])) > 0.999):
            strnEle = np.matmul(sdStrnMode['value'][:, -5:-2], sdIntgConst[-5:-2])  # Eq. (3.64)
        else:  # stress singularity at scaling centre
            strnEle = [float('nan')] * len(sdStrnMode['value'])

    # remove possible tiny imaginary part due to numerical error
    dsp = np.real(dsp)
    strnEle = np.real(strnEle)

    # strnEle(1:3,ie) is the strains at centre of element ie after reshaping.
    strnEle = np.reshape(strnEle, (3, -1), order='F')

    # nodal stresses by averaging element stresses
    nNode = len(sdSln['node'])  # number of nodes
    strnNode = np.zeros((3, nNode))  # initialisation
    # counters of number of elements connected to a node
    count = np.zeros(nNode)
    n = sdSln['conn'][:, 0]  # vector of first node of elements
    # add element stresses to first node of elements
    strnNode[:, n] += strnEle
    # increment counters
    count[n] += 1

    n = sdSln['conn'][:, 1]  # vector of second node of elements
    # add element stresses to second node of elements
    strnNode[:, n] += strnEle
    # increment counters
    count[n] += 1
    strnNode = strnNode / count  # averaging


    return (nodexy, dsp, strnNode, GPxy, strnEle)
