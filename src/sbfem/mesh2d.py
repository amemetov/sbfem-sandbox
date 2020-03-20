from enum import Enum

class MesherType(Enum):
    DistMesh = 1
    PolyMesher = 2
    DirectTriangMesh = 3
    DirectPolygonMesh = 4


def createSBFEMesh(mesherType:MesherType):
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
