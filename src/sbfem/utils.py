import numpy as np
import matplotlib.pyplot as plt


def matlabToPythonIndices(indices):
    """
    Converts matlab indices (starting from 1) to corresponding python indices (starting from 0).
    Expects one of the following types: `int`, `list` or `numpy array`.
    For other cases raises ValueError.
    """
    if isinstance(indices, int):
        return indices - 1

    if isinstance(indices, list):
        return [matlabToPythonIndices(i) for i in indices]

    if isinstance(indices, np.ndarray):
        return indices - 1

    raise ValueError("`matlabToPythonIndices` allows only one of the following types of argument: "
                     f"`int`, `list`, `numpy array`. Got {type(indices)}")

def plotSBFEMesh(coord, sdConn, opt):
    """
    The original name: PlotSBFEMesh (p.98)
    Plot polygon S-element mesh
    :param coord: coord(i,:)   - coordinates of node i
    :param sdConn: sdConn{isd}(ie,:) - S-element conncetivity. The nodes of line element ie in S-element isd.
    :param opt: plot options a dict where the options are:
        sdSC: scaling centers of S-elements.
        LabelSC: If specified, plot a marker at the scaling centre = 0,
                do not label S-element > 0,
                show S-element number If > 2, it also specifies the font size.
        fill: =[r g b]. Fill an S-element with color. sdSC has also to be given.
        LineSpec: LineSpec of 'plot' function in Matlab
        LineWidth: LineWidth of 'plot' function in Matlab
        PlotNode: = 0, do not plot node symbol; otherwise, plot
        LabelNode: = 0, do not label nodes; > 0, show nodal number.
                    LabelNode > 5, it specifies the font size.
                    < 0, draw a marker only
        MarkerSize: marker size of nodes and scaling centres
        BC_Disp: if specified, plot a marker at a node with prescribed displacement(s)
        BC_Frc: if specified, plot a marker at a node with applied force(s)
    """

    if 'title' in opt:
        plt.title(opt['title'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')

    LineWidth = 1
    LineSPec = '-'
    # use specified LineSpec if present
    if opt is not None:
        if 'LineSpec' in opt and len(opt['LineSpec']) > 0:
            LineSPec = opt['LineSpec']
        if 'LineWidth' in opt and len(opt['LineWidth']) > 0:
            LineWidth = opt['LineWidth']

    nsd = len(sdConn)  # number of S-elements

    # plot mesh
    meshEdge = np.empty((0, 2), dtype=np.int)
    for isd in range(nsd):
        meshEdge = np.vstack((meshEdge, sdConn[isd]))

    # sort by rows
    meshEdge = np.sort(meshEdge, axis=1)
    meshEdge = np.unique(meshEdge, axis=0)

    X = np.vstack((coord[meshEdge[:, 0], 0].T, coord[meshEdge[:, 1], 0].T))
    Y = np.vstack((coord[meshEdge[:, 0], 1].T, coord[meshEdge[:, 1], 1].T))
    plt.plot(X, Y, color='green', linestyle=LineSPec, linewidth=LineWidth)

    if 'savePath' in opt:
        plt.savefig(opt['savePath'])


def plotDeformedMesh(d, coord, sdConn, opt):
    """
    The original name: PlotDeformedMesh (p.101)
    :param d: nodal displacements
    :param coord: coord(i,:)   - coordinates of node i
    :param sdConn: sdConn{isd}(ie,:) - S-element conncetivity. The nodes of line element ie in S-element isd.
    :param opt: a dict with a keys:
                MagnFct: magnification factor of deformed mesh
                Undeformed : style of undeformed mesh
    """
    if opt is None:
        opt = dict()

    magnFct = opt['MagnFct'] if 'MagnFct' in opt else 0.1

    # maximum displacement
    Umax = np.max(np.abs(d))
    # maximum dimension of domain
    Lmax = np.max(np.max(coord)-np.min(coord))
    # factor to magnify the displacement
    fct = magnFct * Lmax/Umax
    # augment nodal coordinates
    deformed = coord + fct * np.reshape(d, (2, -1), order='F').T

    # plot undeformed mesh
    if 'Undeformed' in opt and len(opt['Undeformed']) > 0:
        # plotting option of undeformed mesh
        undeformedopt = {'LineSpec': opt['Undeformed']}
        plotSBFEMesh(coord, sdConn, undeformedopt)

    plt.title('DEFORMED MESH')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')

    # plot deformed mesh
    deformedopt = {'LineSpec': '-'}  # plotting option
    plotSBFEMesh(deformed, sdConn, deformedopt)

    if 'savePath' in opt:
        plt.savefig(opt['savePath'])

    return deformed, fct
