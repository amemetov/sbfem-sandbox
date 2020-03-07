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


def plotMultipleText(X, Y, text, **kwargs):
    for x, y, t in zip(X, Y, text):
        plt.text(x, y, t, kwargs)


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
        if 'LineSpec' in opt and opt['LineSpec'] is not None:
            LineSPec = opt['LineSpec']
        if 'LineWidth' in opt and opt['LineWidth'] is not None:
            LineWidth = opt['LineWidth']

    nsd = len(sdConn)  # number of S-elements

    meshEdge = np.empty((0, 2), dtype=np.int)
    for isd in range(nsd):
        meshEdge = np.vstack((meshEdge, sdConn[isd]))

    # TODO: implement
    # # fill S-elements by treating scaling centre and an edge as a triangle
    # if 'sdSC' in opt and opt['sdSC']  is not None:
    #     if 'fill' in opt and opt['fill'] is not None:
    #         p = np.vstack((coord, opt['sdSC']))  # points
    #         nNode = coord.shape[0]  # number of nodes
    #         #  appending scaling centres
    #         nEdge = cellfun(@length, sdConn)
    #         #  initilisation of array of scaling centre
    #         cnt = np.zeros(sum(nEdge))
    #         ib = 1  #  starting index
    #         for ii in range(nsd):
    #             ie = ib-1 + nEdge[ii]  #  ending index
    #             cnt[ib:ie] = nNode + ii  #  scaling centre
    #             ib = ie + 1;
    #         t = [meshEdge cnt]; #  triangles
    #         patch('Faces',t,'Vertices',p, 'FaceColor',opt.fill,'LineStyle','none');

    # plot mesh
    meshEdge = np.sort(meshEdge, axis=1)  # sort by rows
    meshEdge = np.unique(meshEdge, axis=0)

    X = np.vstack((coord[meshEdge[:, 0], 0].T, coord[meshEdge[:, 1], 0].T))
    Y = np.vstack((coord[meshEdge[:, 0], 1].T, coord[meshEdge[:, 1], 1].T))
    plt.plot(X, Y, LineSPec, linewidth=LineWidth)

    # apply plot options
    if 'MarkerSize' in opt:
        markersize = opt['MarkerSize']
    else:
        markersize = 5
    if 'sdSC' in opt and opt['sdSC'] is not None:
        if 'LabelSC' in opt and opt['LabelSC'] is not None:
            # plot scaling centre
            plt.plot(opt['sdSC'][:, 0], opt['sdSC'][:, 1], 'r+', markersize=markersize, linewidth=LineWidth)
            plt.plot(opt['sdSC'][:, 0], opt['sdSC'][:, 1], 'ro', fillstyle='none', markersize=markersize, linewidth=LineWidth)
            if opt['LabelSC'] > 1:
                if opt['LabelSC'] > 5:
                    fontsize = opt['LabelSC']
                else:
                    fontsize = 12
                plotMultipleText(opt['sdSC'][:, 0], opt['sdSC'][:, 1], [' ' + str(x) for x in np.arange(nsd)], color='r', fontsize=fontsize)
    if 'PlotNode' in opt and opt['PlotNode']:
        # showing nodes by plotting a circle
        plt.plot(coord[:, 0], coord[:, 1], 'ko', fillstyle='none', markersize=markersize, linewidth=LineWidth)
    if 'LabelNode' in opt and opt['LabelNode']:
        nNode = coord.shape[0]
        if opt['LabelNode'] > 1:
            fontsize = opt['LabelNode']
        else:
            fontsize = 12
        # showing nodes by plotting a circle
        plt.plot(coord[:, 0], coord[:, 1], 'ko', fillstyle='none', markersize=markersize, linewidth=LineWidth)
        # label nodes with nodal number
        plotMultipleText(coord[:, 0], coord[:, 1], [' ' + str(x) for x in np.arange(nNode)], fontsize=fontsize)
    if 'BC_Disp' in opt and opt['BC_Disp'] is not None:
        # show fixed DOFs by a marker at the nodes
        Node = opt['BC_Disp'][:, 0]
        plt.plot(coord[Node, 0], coord[Node, 1], 'b>', fillstyle='none', markersize=8, linewidth=LineWidth)
    if 'BC_Frc' in opt and opt['BC_Frc'] is not None:
        # show DOFs carrying external forces by a marker at the nodes
        Node = opt['BC_Frc'][:, 0]
        plt.plot(coord[Node, 0], coord[Node, 1], 'm^', fillstyle='none', markersize=8, linewidth=LineWidth);

    if 'savePath' in opt:
        plt.savefig(opt['savePath'])

    if 'show' in opt and opt['show']:
        plt.show()


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
    deformedopt = {'LineSpec': 'k-'}  # plotting option
    plotSBFEMesh(deformed, sdConn, deformedopt)

    if 'savePath' in opt:
        plt.savefig(opt['savePath'])

    if 'show' in opt and opt['show']:
        plt.show()

    return deformed, fct


def plotStressContour(X, Y, C, levels=None, cmap='jet', opt=None):
    if opt is None:
        opt = dict()

    cs = plt.contourf(X, Y, C, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cs)

    if 'savePath' in opt:
        plt.savefig(opt['savePath'])

    if 'show' in opt and opt['show']:
        plt.show()

    plt.show()