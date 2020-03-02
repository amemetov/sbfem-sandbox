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
    end
end

