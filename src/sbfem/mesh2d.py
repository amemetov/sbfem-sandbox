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
        coord(i,:)  -  coordinates of node i
        sdConn{isd,:}(ie,:)  - S-element conncetivity. The nodes of line element ie in S-element isd.
        sdSC(isd,:)  - coordinates of scaling centre of S-element isd
    """
    pass
