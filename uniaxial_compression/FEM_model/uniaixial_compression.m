%md# Uniaxial Compression example
%md---
%md
clear all, close all

% Define input arguments
arg_list = argv ();

if nargin < 3 
  error("At least Lx, Ly and Lz must be defined")
end
Lx = str2num(arg_list{1}) ; Ly = str2num(arg_list{2}) ; Lz = str2num(arg_list{3}) ;

if nargin >= 4
  E = str2num(arg_list{4}) ;
else
  E =1;
end

if nargin >= 5
  nu = str2num(arg_list{5}) ;
else
  nu = .3;
end

if nargin >= 6
  p = str2num(arg_list{6}) ;
else
  p = 3;
end

function [matUs,loadFactorsMat,analyticVals] = uniaxialCompression(Lx,Ly,Lz,E,nu,p)
  %md
  addpath( genpath( getenv("ONSAS_PATH") ) );
  %md
  %md### MEBI parameters
  %md
  %md#### materials
  %md The material of the solid considered is the Saint-Venant-Kirchhoff with Lamé parameters computed as
  lambda = E*nu/((1+nu)*(1-2*nu)) ; mu = E/(2*(1+nu)) ;
  %md since only one material is considered, a scalar struct is defined as follows
  materials.hyperElasModel = 'NHC' ;
  bulk = E / ( 3*(1-2*nu) ) ;
  materials.hyperElasParams = [ mu bulk ] ;
  %md
  %md#### elements
  %md In this model two kinds of elements are used: `tetrahedron` for the solid and `triangle` for introducing the external loads. Since two kinds of elements are used, the struct have length 2:
  elements(1).elemType = 'triangle' ;
  elements(2).elemType = 'tetrahedron' ;
  elements(2).elemTypeParams = [ 2 ] ;
  %md
  %md#### boundaryConds
  %md in this case four BCs are considered, one corresponding to a load and three to displacements.
  %md the first BC introduced is a load, then the coordinate system, loadfactor time function and base load vector are defined
  boundaryConds(1).loadsCoordSys = 'global';
  boundaryConds(1).loadsTimeFact = @(t) p*t ;
  boundaryConds(1).loadsBaseVals = [ -1 0 0 0 0 0 ] ;
  %md the other BCs have imposed displacements
  boundaryConds(2).imposDispDofs = [1] ;
  boundaryConds(2).imposDispVals =  0  ;
  %
  boundaryConds(3).imposDispDofs = [3] ;
  boundaryConds(3).imposDispVals =  0  ;
  %
  boundaryConds(4).imposDispDofs = [5] ;
  boundaryConds(4).imposDispVals =  0  ;
  %
  %md
  %md#### initialConds
  %md since no initial non-homogeneous initial conditions are used, an empty struct is used .
  initialConds = struct();
  %md
  %md### Mesh
  %md A simple hand-made 8-node mesh, with 6 tetrahedrons is considered
  %md
  %md```@raw html
  %md<img src="https://raw.githubusercontent.com/ONSAS/ONSAS_docs/master/docs/src/solidCubeMeshHTML.svg" alt="mesh diagram" width="500"/>
  %md```
  %md```@raw latex
  %md\begin{center}
  %md\def\svgwidth{0.6\textwidth}
  %md\input{solidCubeMeshPDF.pdf_tex}
  %md\end{center}
  %md```
  %md The node coordinates matrix is given by the following
  mesh.nodesCoords = [ 0    0    0 ; ...
                      0    0   Lz ; ...
                      0   Ly   Lz ; ...
                      0   Ly    0 ; ...
                      Lx   0    0 ; ...
                      Lx   0   Lz ; ...
                      Lx  Ly   Lz ; ...
                      Lx  Ly    0 ] ;
  %md and the connectivity cell is defined as follows with the four MEBI parameters for each element followed by the indexes of the nodes of each element. All the eight triangle elements are considered with no material (since they are used only to include load) and the following six elements are solid SVK material tetrahedrons.
  mesh.conecCell = {[ 0 1 1    5 8 6   ]; ... % loaded face
                    [ 0 1 1    6 8 7   ]; ... % loaded face
                    [ 0 1 2    4 1 2   ]; ... % x=0 supp face
                    [ 0 1 2    4 2 3   ]; ... % x=0 supp face
                    [ 0 1 3    6 2 1   ]; ... % y=0 supp face
                    [ 0 1 3    6 1 5   ]; ... % y=0 supp face
                    [ 0 1 4    1 4 5   ]; ... % z=0 supp face
                    [ 0 1 4    4 8 5   ]; ... % z=0 supp face
                    [ 1 2 0    1 4 2 6 ]; ... % tetrahedron
                    [ 1 2 0    6 2 3 4 ]; ... % tetrahedron
                    [ 1 2 0    4 3 6 7 ]; ... % tetrahedron
                    [ 1 2 0    4 1 5 6 ]; ... % tetrahedron
                    [ 1 2 0    4 6 5 8 ]; ... % tetrahedron
                    [ 1 2 0    4 7 6 8 ]  ... % tetrahedron
                  } ;

  %md
  %md### Analysis parameters
  %md
  analysisSettings.methodName    = 'newtonRaphson' ;
  analysisSettings.stopTolIts    = 30     ;
  analysisSettings.stopTolDeltau = 1.0e-8 ;
  analysisSettings.stopTolForces = 1.0e-8 ;
  analysisSettings.finalTime      = 1      ;
  analysisSettings.deltaT        = .1     ;
  %md
  %md### Output parameters
  otherParams.plots_format = 'vtk' ;
  otherParams.problemName = 'uniaxial_compression' ;
  %md
  [matUs, loadFactorsMat] = ONSAS( materials, elements, boundaryConds, initialConds, mesh, analysisSettings, otherParams ) ;
  %md



  alphas         = (Lx + matUs(6*6+1,:)) / Lx ;
  betas          = (Ly + matUs(6*6+3,:)) / Ly ;

  analyticFunc = @(alphas,betas) mu * alphas - mu*1./alphas + bulk * alphas .* betas.^2 .* ( alphas .* betas.^2 -1) ./ alphas ;
  analyticVals = analyticFunc( alphas, betas ) ;

end


[matUs,loadFactorsMat, analyticVals] = uniaxialCompression(Lx,Ly,Lz,E,nu,p)


controlDispsValsCase = matUs(6*6+1,:) ;  
%disp(matUs)
matUslast = matUs(:,end)
loadedfacenodeindexes = [ 5 6 7 8 ] ;
loadedfaceDoFs = nodes2dofs( loadedfacenodeindexes, 6 ) ;
loadedfaceDoFsUs = loadedfaceDoFs(1:2:end);
Usloaded=matUslast(loadedfaceDoFsUs);
Ux = mean(Usloaded(1:3:end));
Uy = (Usloaded(8)+Usloaded(11))/2;
Uz = (Usloaded(6)+Usloaded(9))/2;
%return Ux, Uy, Uz;
%dlmwrite('output.txt', [Ux, Uy, Uz]);
fid = fopen('output.txt', 'w');
fprintf(fid, '%f\n', Ux);
fprintf(fid, '%f\n', Uy);
fprintf(fid, '%f\n', Uz);
fclose(fid);
%md## Plot
%mdThe numerical and analytic solutions are plotted.
% lw = 2.0 ; ms = 11 ; plotfontsize = 18 ;
% figure, hold on, grid on
% plot( controlDispsValsCase, loadFactorsMat, 'r-x' , 'linewidth', lw,'markersize',ms )
% plot( controlDispsValsCase, analyticVals,  'g-s' , 'linewidth', lw,'markersize',ms )
% labx = xlabel('Displacement');   laby = ylabel('\lambda(t)') ;
% legend( 'Numeric', 'Analytic' , 'location', 'SouthEast' )
% set(gca, 'linewidth', 1.0, 'fontsize', plotfontsize )
% set(labx, 'FontSize', plotfontsize); set(laby, 'FontSize', plotfontsize) ;
% print("./output/validation.png")
