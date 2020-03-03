classdef Ch3
    methods(Static)
        function [K, d, v, M] = example_3_1()
            %% Solution of a square S-element
            xy = [-1 -1; 1 -1; 1 1; -1 1]
            conn = [1:4 ; 2:4 1]'
            % elascity matrix (plane stress).
            % E: Young’s modulus; p: Poisson’s ratio
            ElasMtrx = @(E, p) E/(1-p^2)*[1 p 0; p 1 0; 0 0 (1-p)/2];
            mat.D = ElasMtrx(10, 0); % E in GPa
            mat.den = 2; % mass density in Mg per cubic meter
            [ E0, E1, E2, M0 ] = SElementCoeffMtx(xy, conn, mat)
            [ K, d, v, M ] = SElementSlnEigenMethod(E0, E1, E2, M0)
            %disp(v)
        end

        function [K, d, v, M] = example_3_2()
            %% Solution of pentagon S-element
            xy = [cosd(-126:72:180); sind(-126:72:180)]';
            conn = [1:5; 2:5 1]';
            % elasticity matrix (plane stress).
            % E: Young’s modulus; p: Poisson’s ratio
            ElasMtrx = @(E, p) E/(1-p^2)*[1 p 0; p 1 0; 0 0 (1-p)/2];
            mat.D = ElasMtrx(10, 0); % E in GPa
            mat.den = 2; % mass density in Mg per cubic meter
            [ E0, E1, E2, M0 ] = SElementCoeffMtx(xy, conn, mat);
            [ K, d, v, M ] = SElementSlnEigenMethod(E0, E1, E2, M0)
            %disp(v)
        end
        
        function [sdSln, K, M] = example_3_3()
            % input problem definition
            Exmpl3SElements()
            % solution of S-elements and assemblage of global stiffness and mass matrices
            [sdSln, K, M] = SBFEMAssembly(coord, sdConn, sdSC, mat);
        end
        
        function [] = example_3_4()
            close all; clearvars; dbstop error;
            
            Exmpl3SElements();
            
            SBFEM2DStaticAnalysisScript;
            dofs = (1:numel(coord))';
            format long g;
            [dofs d(dofs) F(dofs)]            
        end
        
        function [] = example_3_5()
            close all; clearvars; dbstop error;
            
            ExmplDeepBeam();
            
            SBFEM2DStaticAnalysisScript;
            
            dofs = 2 * [3 7 12 15 16]
            [dofs d(dofs)]            
        end
        
        function [] = example_3_6()
            close all; clearvars; dbstop error;
            
            ExmplEdgeCrackedRectangularPlate;
            
            SBFEM2DStaticAnalysisScript;
            
            NodalDisp = (reshape(d, 2, []))';
            disp(' Crack opening displacement');
            COD = NodalDisp(17, :) - NodalDisp(1, :);
            disp(COD)
        end
        
        function [] = example_3_7()
            close all; clearvars; dbstop error;
            
            Exmpl3SElements();
            
            SBFEM2DStaticAnalysisScript;
            dofs = (1:numel(coord))';
            format long g;
            [dofs d(dofs) F(dofs)]
            
            % strain modes of S-elements
            sdStrnMode = SElementStrainMode2NodeEle( sdSln );
            % integration constants
            sdIntgConst = SElementIntgConst( d, sdSln );
            
            isd = 2; % S-element number
            
            % display integration constants
            [(1:length(sdIntgConst{isd}))' sdIntgConst{isd}]
            disp('strain modes')
            sdStrnMode{isd}.value
            
            xi = 0.5; % radial coordinate
            % displacements and strains at specified raidal coordinate
            [nodexy, dsp, strnNode, GPxy, strnEle] = ...
            SElementInDispStrain(xi, sdSln{isd}, ...
            sdStrnMode{isd}, sdIntgConst{isd});
            
            disp([' x        y            ux            uy']);
            [nodexy, (reshape(dsp,2,[]))']            
            
            disp('strains of Elements 1 and 2')
            strnEle(:,:)
            
            disp('stresses of Elements 1 and 2')
            mat.D*strnEle(:,:)
        end

        function [] = example_3_8()
            close all; clearvars; dbstop error;

            % Mesh
            % nodal coordinates. One node per row [x y]
            x1 = 0.5; y1 = 0.5; % Figure b
            % x1 = 0.05; y1 = 0.95; % Figure c
            coord = [ x1 y1; 0 0; 0.1 0; 1 0; 1 1; 0 1; 0 0.1];
            % Input S-element connectivity as a cell array (One S-element per cell).
            % In a cell, the connectivity of line elements is given by one element per row
            % [Node-1 Node-2].
            sdConn = { [1 7; 7 2; 2 3; 3 1]; % S-element 1
            [1 3; 3 4; 4 5; 5 6; 6 7; 7 1]}; % S-element 2

            % coordinates of scaling centres of S-elements.
            if x1 > y1 % extension of line 21 intersecting right edge of the square
                sdSC = [ x1/2 y1/2 ; (1+x1)/2 y1*(1+x1)/(2*x1)];
            else %extension of line 21 intersecting top edge of the square
                sdSC = [ x1/2 y1/2 ; x1*(1+y1)/(2*y1) (1+y1)/2];
            end

            % Materials
            mat.D = IsoElasMtrx(1, 0.25); % elasticity matrix
            mat.den = 2; % mass density

            % Boundary conditions
            % displacement constrains. One constrain per row: [Node Dir Disp]
            BC_Disp = [2 1 0; 2 2 0; 4 2 0];
            % assemble load vector
            ndn = 2; % 2 DOFs per node
            NDof = ndn*size(coord,1); % number of DOFs
            F = zeros(NDof,1); % initializing right-hand side of equation [K]{u} = {F}

            % horizontal tension
            % edge = [ 4 5; 6 7; 7 2];
            % trac = [1 0 1 0; -1 0 -1 0; -1 0 -1 0]’;
            % vertical tension
            % edge = [2 3; 3 4; 5 6];
            % trac = [0 -1 0 -1; 0 -1 0 -1; 0 1 0 1]’;
            % pure shear
            % edges subject to tractions, one row per edge
            edge = [2 3; 3 4; 4 5; 5 6; 6 7; 7 2];
             % tractions, one column per edge
            trac = [-1 0 -1 0; -1 0 -1 0; 0 1 0 1; 1 0 1 0; 0 -1 0 -1; 0 -1 0 -1]';
            F = addSurfTraction(coord, edge, trac, F);

            % Plot mesh
            figure
            % plotting options
            opt=struct('LineSpec','-k', 'sdSC',sdSC, 'PlotNode',1, 'LabelNode', 1);
            PlotSBFEMesh(coord, sdConn, opt);
            title('MESH');

            % Static solution
            % nodal displacements
            SBFEM2DStaticAnalysisScript
            disp('Nodal displacements')
            for ii = 1:length(coord)
                fprintf('%5d %25.15e %25.15d\n',ii, d(2*ii-1:2*ii))
            end

            % Stresses
            % strain modes of S-elements
            sdStrnMode = SElementStrainMode2NodeEle( sdSln );
            % integration constants
            sdIntgConst = SElementIntgConst( d, sdSln );
            % displacements and strains at specified radial coordinate
            isd = 2; % S-element number
            xi = 1; % radial coordinate
            [nodexy, dsp, strnNode, GPxy, strnEle] = SElementInDispStrain(xi, sdSln{isd}, sdStrnMode{isd}, sdIntgConst{isd});

            disp('Stresses of Elements 1 and 2')
            mat.D*strnEle(:,1:2)
        end
    end
end

