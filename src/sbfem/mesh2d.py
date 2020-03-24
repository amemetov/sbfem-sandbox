from enum import Enum
import numpy as np


class MesherType(Enum):
    DistMesh = 1
    PolyMesher = 2
    DirectTriangMesh = 3
    DirectPolygonMesh = 4


def createSBFEMesh(mesherType: MesherType):
    """
    Generate polygon S-element mesh
    :return: a tuple (coord, sdConn, sdSC) where:
        coord(i,:) -  coordinates of node i
        sdConn{isd,:}(ie,:) - S-element conncetivity. The nodes of line element ie in S-element isd.
        sdSC(isd,:) - coordinates of scaling centre of S-element isd
    """
    pass


def triToSBFEMesh(p, t):
    """
    Convert a triangular mesh to an S-element mesh
    :param p: p(i,:) - coordinates of node i
    :param t: t(i,:) - nodal numbers of triangle i
    :return: coord, sdConn, sdSC where:
        coord(i,:) - coordinates of node i
        sdConn{isd,:}(ie,:) - an S-element connectivity. The nodes of line element ie in S-element isd.
        sdSC(isd,:) - coordinates of scaling centre of S-element isd
    """
    np = len(p)  # number of points
    nTri = len(t)  # number of triangles
    # centriods of triangles will be nodes of S-elements.
    # triangular element numbers will be the nodal number of S-elements.
    triCnt = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 3], :]) / 3  # centroids

    # construct data on mesh connectivity
    meshEdge, _, edge2sd, node2Edge, node2sd = meshConnectivity(t)


def meshConnectivity(sdConn):
    """
    Construct mesh connectivity data
    The input to this function, sdConn, is the element connectivity of a triangular, polygon or S-element mesh.
    :param sdConn: S-element/element connectivity
            when sdConn is a matrix, sdConn(i,:) are the nodes of element i (triangular)
            when sdConn is a cell array of a 1D array, sdConn{i} are the nodes of polygon i (polygon)
            when sdConn is a cell array of a matrix, sdConn{i} are the nodes of line elements of S-element i (S-element mesh)
    :return: a tuple (meshEdge, sdEdge, edge2sd, node2Edge, node2sd) where:
        meshEdge(i,1:2)  - the 2 nodes on line i in a mesh.
                            The node number of the first node is larger than that of the 2nd one
        sdEdge{i}        - lines forming S-element i,
                            >0 when an line follows anti-clockwise direction around the scaling centre.
                            <0 otherwise
        edge2sd{i}       - S-elements connected to edge i
        node2Edge{i}     - edges connected to node i
        node2sd{i}       - S-elements connected to node i
    """

    nsd = len(sdConn)  # number of S-elements/elements

    # construct connectivity of edges to S-elements/elements
    # sdConn is intepreted based on its data type

    meshEdge, sdEdge = None, None

    if np.issctype(sdConn.dtype):
        # sdConn is a matrix, sdConn(i,:) are the nodes of element i
        # sdConn is a matrix (triangular elements)

        # the following loop collects the edges of all elements
        meshEdge = []
        sdEdge = []
        i1 = 1  # counter of edges
        for ii in range(nsd):  # loop over elements
            eNode = sdConn[ii]  # nodes of an element
            i2 = i1 + len(eNode)  # count the edges
            # element edges. Each edge is defined by two nodes on a column
            meshEdge.append(np.vstack((eNode, np.hstack((eNode[1:], eNode[0])))))
            sdEdge.append(np.arange(i1, i2))  # edges of an element
            # store the edges to be reversed in sorting as negative values
            idx = meshEdge[ii][0, :] > meshEdge[ii][1, :]
            sdEdge[ii][idx] = -sdEdge[ii][idx]
            i1 = i2  # update the counter for the next element
        # combine edges of all elements.
        meshEdge = np.hstack(meshEdge[:])
        # The two nodes of an edge are sorted in ascending order.
        meshEdge = np.sort(meshEdge, axis=0)
        # Each edge is stored as one row.
        meshEdge = meshEdge.T
    else:
        sdEdge = []
        i1 = 1
        if np.ndim(sdConn[0]) > 1:
            # sdConn is a cell array of a matrix, sdConn{i} are the nodes of line elements of S-element i
            # S-elements (a cell has multiple rows)
            # all the edges of S-elements are numbered in this loop
            for ii in range(nsd):  # loop over S-elements
                i2 = i1 + len(sdConn[ii])  # count the edges
                sdEdge.append(np.arange(i1, i2))  # edges of an S-element
                # store the edges to be reversed in sorting as negative values
                # each element edge is defined by two nodes as a row
                idx = sdConn[ii][:, 0] > sdConn[ii][:, 1]
                sdEdge[ii][idx] = -sdEdge[ii][idx]
                i1 = i2
            # combine edges of all S-elements.
            meshEdge = np.vstack(sdConn[:])
            # The two nodes of an edge are sorted in ascending order.
            # Each edge is stored as one row.
            meshEdge = np.sort(meshEdge, axis=1)
            # meshEdge = sort(vertcat(sdConn{:}),2);
        else:
            # sdConn is a cell array of a 1D array, sdConn{i} are the nodes of polygon i
            pass

    # remove duplicated entries of edges
    meshEdge, ic = np.unique(meshEdge, axis=0, return_inverse=True)
    for ii in range(nsd):  # loop over S-elements/elements
        # update edge numbers
        # store sign of 1-based indices
        sign = np.sign(sdEdge[ii])
        # convert to 0-based indices
        sdEdge[ii] = np.abs(sdEdge[ii]) - 1
        sdEdge[ii] = sign * ic[np.abs(sdEdge[ii])]

    return meshEdge, sdEdge
