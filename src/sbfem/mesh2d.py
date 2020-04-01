from enum import Enum
import numpy as np
import sbfem.utils as utils


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


def polygonToSBFEMesh(coord, polygon):
    """
    Convert a polygon mesh to an S-element mesh
    :param coord: coord(i,:)   - coordinates of node i
    :param polygon: polygon{i}   - array of vertices of polygon i
    :return: a tuple (sdConn, sdSC) where:
        sdConn{isd,:}(ie,:)  - S-element conncetivity. The nodes of line element ie in S-element isd.
        sdSC(isd,:)  - coordinates of scaling centre of S-element isd.
    """
    # sdConn - S-element connectivity stored as a cell array/list (One S-element per cell).
    # In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2].
    nsd = len(polygon)  # number of S-elements
    sdConn = []  # initialising connectivity
    sdSC = np.zeros((nsd, 2))  # scaling centre
    for isub in range(nsd):
        # build connectivity
        sdConn.append(np.vstack((polygon[isub], np.hstack((polygon[isub][1:], polygon[isub][0])))).T)
        # scaling centre at centroid of polygon
        sdSC[isub] = utils.polygonCentroid(coord[polygon[isub]])

    return sdConn, sdSC


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
    nump = len(p)  # number of points
    nTri = len(t)  # number of triangles
    # centriods of triangles will be nodes of S-elements.
    # triangular element numbers will be the nodal number of S-elements.
    triCnt = (p[t[:, 0], :] + p[t[:, 1], :] + p[t[:, 2], :]) / 3  # centroids

    # construct data on mesh connectivity
    meshEdge, _, edge2sd, node2Edge, node2sd = meshConnectivity(t)

    numEdges = len(edge2sd)

    # number of S-elements connected to an edge
    edgeNsd = np.array([len(s) for s in edge2sd])
    # list of boundary edges (connected to 1 S-element only)
    bEdge = np.argwhere(edgeNsd == 1).flatten()
    # midpoints of boundary edges
    bEdgeCentre = (p[meshEdge[bEdge, 0], :] + p[meshEdge[bEdge, 1], :]) / 2
    # list of points at boundary
    bp = np.unique(meshEdge[bEdge])

    # include the points in the middle of boundary edges as nodes of S-elements
    nbEdge = len(bEdge)
    bEdgeNode = nTri + np.arange(0, nbEdge).T  # nodal number
    # index from edge number
    bEdgeIdx = np.full((numEdges,), -1)
    bEdgeIdx[bEdge] = np.arange(0, nbEdge)

    # include the points at boundary as nodes of S-element mesh
    nbp = len(bp)
    bNode = nTri + nbEdge + np.arange(0, nbp).T  # nodal number
    # index from point number
    bpIdx = np.full((nump,), -1)
    bpIdx[bp] = np.arange(0, nbp)
    # nodal coordinates
    coord = np.vstack((triCnt, bEdgeCentre, p[bp]))

    # construct polygon S-elements
    sdConn = []  # initializing connectivity
    sdSC = []  # initializing scaling centre

    for ii in range(nump):
        if bpIdx[ii] == -1:  # interior point
            node = node2sd[ii]  # S-elements connected to node ii
            # sort nodes in counter-clockwise direction
            node = sortNodes(node, coord[node], p[ii])
            # scaling centre at current point
            sdSC.append(p[ii])
            # line element connectivity in an S-element
            sdConn.append(np.vstack((node, np.hstack((node[1:], node[0])))).T)
        else:  # boundary point, which can become a node or a scaling centre
            be = bEdgeIdx[node2Edge[ii]]  # edges connected to node
            nodee = bEdgeNode[be[be != -1]]  # nodes on boundary edges
            # sort the nodes, except for the one at the current point
            node = np.concatenate((node2sd[ii], nodee))
            node = sortNodes(node, coord[node], p[ii])

            # find the two boundary nodes in the node list
            idx1 = np.argwhere(node == nodee[0])[0][0]
            idx2 = np.argwhere(node == nodee[1])[0][0]

            # maintain counter-clockwise direction and rearrange the nodes as:
            # [boundary node 1, current point (node), boundary node 2, others]
            if abs(idx1 - idx2) > 1:
                # the two boundary nodes are the first and last in the list
                node = np.concatenate((node[-1:], bNode[bpIdx[ii]:bpIdx[ii] + 1], node[0:-1]))
            else:
                # the two boundary nodes are consecutive on the list
                idx = min(idx1, idx2)
                node = np.concatenate((node[idx:idx + 1], bNode[bpIdx[ii]:bpIdx[ii] + 1], node[idx + 1:], node[0:idx]))

            # internal angle between two boundary edges
            dxy = np.diff(coord[node[0:3]], axis=0)  # Δx, Δy of the 1st 2 edge
            dl = np.sqrt(np.sum(dxy ** 2, axis=1))  # length
            dxyn = dxy / dl[:, np.newaxis]  # direction cosin
            # angle between 2 boundary edges
            alpha = np.real(np.arccos(np.sum(dxyn[0, :] * dxyn[1, :])))
            beta = 180 - np.sign(np.linalg.det(dxyn)) * alpha  # internal angle
            if beta < 220:  # include current point as a node
                # line element connectivity in an S-element
                sdConn.append(np.vstack((node, np.hstack((node[1:], node[0])))).T)
                # select centroid as scaling centre
                sdSC.append(utils.polygonCentroid(coord[node]))
            else:  # use current point (concave corner) as a scaling centre
                sdSC.append(p[ii])
                # line element connectivity in an S-element
                sdConn.append(np.vstack((node[2:], np.hstack((node[3:], node[0])))).T)

    # remove unconnected nodes
    a = np.ravel(np.concatenate(sdConn))  # all nodes
    c = np.unique(a)  # unique nodes
    i = np.zeros_like(a)
    i[c] = np.arange(0, len(c))  # new nodal numbers of the unique nodes
    coord = coord[c]  # update the nodal coordinates accordingly
    # update line element connectivity in each S-element
    for ii in range(len(sdConn)):
        sdConn[ii] = np.reshape(i[sdConn[ii]], (-1, 2))

    return coord, np.array(sdConn), np.array(sdSC)


def sortNodes(node, xy, c):
    # sort nodes in counterclock direction around point c
    xy = xy - c
    ang = np.arctan2(xy[:, 1], xy[:, 0])  # angular coordinates
    ic = np.argsort(ang)  # sort to increasing angular coordinates
    node = node[ic].T  # rearrange nodes
    return node


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

    # try to convert sdConn to ndarray
    if not isinstance(sdConn, np.ndarray):
        sdConn = np.array(sdConn)

    nsd = len(sdConn)  # number of S-elements/elements

    # ##### construct connectivity of edges to S-elements/elements #####
    # sdConn is interpreted based on its data type

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
            # polygon element (closed loop specified by vertices)
            # the following loop collects the edges of all polygons
            meshEdge = []
            for ii in range(nsd):  # loop over polygons
                eNode = sdConn[ii]  # nodes of a polygon
                i2 = i1 + len(eNode)  # count the edges
                # each element edge is defined by two nodes as a column
                meshEdge.append(np.vstack((eNode, np.hstack((eNode[1:], eNode[0])))))
                sdEdge.append(np.arange(i1, i2))  # edges of a polygon
                idx = meshEdge[ii][0, :] > meshEdge[ii][1, :]
                # edge to be reversed
                sdEdge[ii][idx] = -sdEdge[ii][idx]
                i1 = i2  # update the counter for the next element
            # combine edges of all elements.
            meshEdge = np.hstack(meshEdge[:])
            # The two nodes of an edge are sorted in ascending order.
            meshEdge = np.sort(meshEdge, axis=0)
            # Each edge is stored as one row.
            meshEdge = meshEdge.T

    # remove duplicated entries of edges
    meshEdge, ic = np.unique(meshEdge, axis=0, return_inverse=True)
    for ii in range(nsd):  # loop over S-elements/elements
        # update edge numbers
        # store sign of 1-based indices
        sign = np.sign(sdEdge[ii])
        # convert to 0-based indices
        sdEdge[ii] = np.abs(sdEdge[ii]) - 1
        sdEdge[ii] = sign * ic[np.abs(sdEdge[ii])]

    # ##### find S-elements/elements connected to an edge #####
    a = np.abs(np.concatenate(sdEdge))  # edges of all S-elements/elements
    # the following loop matches S-element/element numbers to edges
    asd = np.zeros_like(a)  # initialization
    ib = 0  # pointer to the first edge of an S-element/element
    for ii in range(nsd):  # loop over S-elements/elements
        ie = ib + len(sdEdge[ii])
        asd[ib:ie] = ii  # edge a(i) is connected to S-element/element asd(i)
        ib = ie  # update the pointer
    # sort S-element numbers according to edge number
    c, indx = np.sort(a), np.argsort(a)
    asd = asd[indx]
    # the following loop collects the S-elements connected to nodes
    ib = 0  # pointer to the 1st S-element/element connected to an edge
    nMeshedges = len(meshEdge)  # number of edges in mesh
    edge2sd = []  # initialization
    for ii in range(nMeshedges - 1):  # loop over edges (except for the last one)
        if c[ib + 1] == ii:  # two S-elements/elements connected to an edge
            edge2sd.append(asd[ib:ib + 2])  # store the S-elements/elements
            ib = ib + 2  # update the pointer
        else:  # one S-element/element connected to an edge
            edge2sd.append(np.atleast_1d(asd[ib]))  # store the S-element/element
            ib = ib + 1  # update the pointer
    # the S-elements/elements connected to the last edges
    edge2sd.append(asd[ib:])
    # sort S-elements for convenience of testing/using
    edge2sd = [np.sort(s) for s in edge2sd]

    # ##### find edges connected to a node #####
    a = np.ravel(meshEdge.T)  # np.reshape(meshEdge.T, ())  # nodes on edges
    # edge numbers corresponding to nodes in a
    edgei = np.ravel(np.tile(np.arange(0, len(meshEdge)), (2, 1)))
    # sort edge number according to node number
    c, indx = np.sort(a), np.argsort(a)
    edgei = edgei[indx]
    ib = 0  # pointer to the 1st edge connected to a node
    nNode = c[-1] + 1  # number of nodes
    node2Edge = []  # initialization
    for ii in range(nNode - 1):  # loop over nodes (except for the last one)
        # pointer to the last edge connected to a node
        ie = ib + np.argwhere(c[ib:] != ii)[0][0]
        node2Edge.append(edgei[ib:ie])  # store edges connected to node
        ib = ie  # + 1  # update the pointer
    # store edges connected to the last node
    node2Edge.append(edgei[ib:])
    # sort edges for convenience of testing/using
    node2Edge = [np.sort(e) for e in node2Edge]

    # ##### find S-elements connected to a node #####
    node2sd = []  # initialization
    np_edge2sd = np.array(edge2sd)  # create to use indexing in the following loop
    for ii in range(nNode):  # loop over nodes
        node2sd.append(np.unique(np.concatenate(np_edge2sd[node2Edge[ii]])))

    return meshEdge, sdEdge, edge2sd, node2Edge, node2sd
