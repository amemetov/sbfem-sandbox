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
        
        function [] = example_3_9()
            clearvars; close all; dbstop if error
            
            %% Input
            R = 1; % radius of circular body
            p = 1000; % radial surface traction (KPa)
            E = 10E6; % E in KPa
            nu = 0.25; % Poisson’s ratio
            den = 2; %mass density in Mg∕m 3
            
            a = 0.75*R; % crack length
            nq = 4; % number of elements on one quadrant
            
            %% Mesh
            % nodal coordinates. One node per row [x, y]
            n = 4*nq; % number of element on boundary
            dangle = 2*pi/n; % angular increment of an element Δ θ
            angle = -pi:dangle:pi+dangle/5; % angular coordinates of nodes
            % Note: there are two nodes at crack mouth with the same coordinates
            % nodal coordinates x = R cos(θ), y = R sin(θ)
            coord = R*[cos(angle);sin(angle)]';
            
            % Input S-element connectivity as a cell array (one S-element per cell). 
            % In a cell, the connectivity of line elements is given by one element per row [Node-1 Node-2].
            sdConn = { [1:n; 2:n+1]' };
            % Note: elements form an open loop
            % select scaling centre at crack tip
            sdSC = [a-R 0];
            
            %% Materials
            % E: Young’s modulus; p: Poisson’s ratio
            ElasMtrx = @(E, p) E/(1-p^2)*[1 p 0;p 1 0;0 0 (1-p)/2];
            mat.D = ElasMtrx(E, nu);
            mat.den = den;
            
            %% Boundary conditions
            % displacement constraints. One constraint per row: [Node Dir Disp]
            BC_Disp = [ nq+1 1 0; 2*nq+1 2 0; 3*nq+1 1 0]; % constrain rigid-body motion
            eleF = R*dangle*p; % resultant radial force on an element pRΔ θ
            % assemble radial nodal forces {F r }
            nodalF = [eleF/2, eleF*ones(1,n-1), eleF/2];
            % nodal forces. One node per row: [Node Dir F]
            % F_x = F_r*cos(θ), F_y = F_r*sin(θ)            
            BC_Frc = [1:n+1 1:n+1; ones(1,n+1) 2*ones(1,n+1); ...
                nodalF.*cos(angle) nodalF.*sin(angle)]';
            
            %% Plot mesh
            h1 = figure;
            opt=struct('LineSpec','-k', 'sdSC',sdSC, ... 
                'PlotNode',1, 'LabelNode', 1,...
                'BC_Disp',BC_Disp); % plotting options
            PlotSBFEMesh(coord, sdConn, opt);
            title('MESH');
            
            % solution of S-elements and global stiffness and mass matrices
            [sdSln, K, M] = SBFEMAssembly(coord, sdConn, sdSC, mat);
            
            %% Assemblage external forces
            ndn = 2; % 2 DOFs per node
            NDof = ndn*size(coord,1); % number of DOFs            
            F = zeros(NDof,1); % initializing right-hand side of equation [K]{u} = {F}
            F = AddNodalForces(BC_Frc, F); % add prescribed nodal forces
            
            %% Static solution
            [U, F] = SolverStatics(K, BC_Disp, F);
            
            CODnorm = (U(end)-U(2))*E/(p*R);
            disp(['Normalised crack openning displacement = ', num2str(CODnorm)])
            
            % plot deformed shape
            Umax = max(abs(U)); % maximum displacement
            fct = 0.2/Umax; % factor to magnify the displacement to 0.2 m argument nodal coordinates
            deformed = coord + fct*(reshape(U,2,[]))';
            hold on
            
            % plotting options
            opt = struct('LineSpec','-ro', 'LabelNode', 1);
            PlotSBFEMesh(deformed, sdConn, opt);
            title('DEFORMED MESH');
            
            %% Internal displacements and stresses
            % strain modes of S-elements
            sdStrnMode = SElementStrainMode2NodeEle( sdSln );
            % integration constants of S-element
            sdIntgConst = SElementIntgConst( U, sdSln );
            
            isd = 1; % S-element number
            xi = (1:-0.01:0).^2; % radial coordinates
            
            % initialization of variables for plotting
            X = zeros(length(xi), length(sdSln{isd}.node));
            Y = X; C = X;
            % displacements and strains at the specified radial coordinate
            for ii= 1:length(xi)
                [nodexy, dsp, strnNode, GPxy, strnEle] = ...
                    SElementInDispStrain(xi(ii), sdSln{isd}, ...
                        sdStrnMode{isd}, sdIntgConst{isd});
                deformed = nodexy + fct*(reshape(dsp,2,[]))';
                % coordinates of grid points
                X(ii,:) = deformed(:,1)';
                Y(ii,:) = deformed(:,2)';
                strsNode = mat.D*strnNode; % nodal stresses
                C(ii,:) = strsNode(2,:); % store σ_yy for plotting
            end
            
            % plot stress contour
            h2 = figure('Color','white');
            contourf(X,Y,C, (-1000:1000:10000), 'LineStyle','none');
            
            hold on
            axis off; axis equal;
            xlabel('x'); ylabel('x');
            colormap(jet);colorbar;
        end
        
        function [] = example_3_10()
            % Shape function of a pentagon S-Element
            clearvars; close all;
            
            % Mesh
            % nodal coordinates
            coord = [cosd(-126:72:180);sind(-126:72:180)]';
            % ... connectivity
            sdConn = { [1:5 ; 2:5 1]' };
            sdSC = [ 0 0]; % one S-Element per row
            
            % Materials
            % E: Young’s modulus; p: Poisson’s ratio
            ElasMtrx = @(E, p) E/(1-p^2)*[1 p 0;p 1 0;0 0 (1-p)/2];
            mat.D = ElasMtrx(10E6, 0.25); % E in KPa
            mat.den = 2; % mass density in Mg∕m 3
            
            % S-Element solution
            [sdSln, K, M] = SBFEMAssembly(coord, sdConn, sdSC, mat);
            
            %%  Shape function
            % prescribed unit displacement
            U = zeros(size(K,1),1);
            U(2*4) = 1;
            
            % strain modes of S-element
            sdStrnMode = SElementStrainMode2NodeEle( sdSln );
            
            % integration constants
            sdIntgConst = SElementIntgConst( U, sdSln );
            
            isd = 1; % S-element number
            xi = 1:-0.01:0; % radial coordinates
            % initialization of variables for plotting
            X = zeros(length(xi), length(sdSln{isd}.node)+1);
            Y = X; Z = X;
            % displacements and strains at the specified radial coordinate
            for ii= 1:length(xi)
                [nodexy, dsp, strnNode, GPxy, strnEle] = ...
                SElementInDispStrain(xi(ii), sdSln{isd}, ...
                sdStrnMode{isd}, sdIntgConst{isd});
                % coordinates of grid points forming a close loop
                X(ii,:) = [nodexy(:,1)' nodexy(1,1)];
                Y(ii,:) = [nodexy(:,2)' nodexy(1,2)];
                Z(ii,:) = [dsp(2:2:end)' dsp(2)]; % store u y for plotting
            end
            
            % ... plot the shape function as a surface
            figure('Color','white')
            surf(X,Y,Z,'FaceColor','interp', 'EdgeColor','none', 'FaceLighting','phong');
            view(-110, 15); % set direction of viewing
            hold on
            text(1.1*(coord(:,1)-0.02), 1.1*coord(:,2), ...
            Z(1,1:end-1)'+0.05,int2str((1:5)')); % label the nodes
            axis equal, axis off;
            xlabel('x'); ylabel('y'); zlabel('N'); % label the axes
            colormap(jet)
            plot3(X(1,:), Y(1,:), Z(1,:), '-b'); % plot edges
            
            % contour of the shape function
            h = figure('Color','white');
            contourf(X,Y,Z,10, 'LineStyle','none'); % 10 contour lines
            hold on
            text(1.05*(coord(:,1)-0.02), 1.05*coord(:,2),int2str((1:5)'));
            axis equal; axis off;
            % show a colourbar indicating the value of the shape function
            colormap(jet)
            caxis([0 1]); colorbar;
            plot(X(1,:), Y(1,:), '-b'); % plot edges
        end
    end
end

