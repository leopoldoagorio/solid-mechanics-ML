%md# Cantilever solid two materials example.
%md---
clear all, close all

% Define input arguments
arg_list = argv ();
arg_list = (Lx, E1, E2, p, nu1, nu2, Ly, Lz)


if nargin < 4 
  error("At least Lx, E1, E2 and p must be defined")
end
Lx = str2num(arg_list{1}) ; 
E1 = str2num(arg_list{2}) ; 
E2 = str2num(arg_list{3}) ;
p = str2num(arg_list{4}) ;

if nargin >= 4
  nu1 = str2num(arg_list{5}) ;
else
  nu1 =.3;
end

if nargin >= 5
  nu2 = str2num(arg_list{6}) ;
else
  nu2 =.3;
end

if nargin >= 6
  Ly = str2num(arg_list{7}) ;
else
  Ly = 1;
end

if nargin >= 7
  Lz = str2num(arg_list{8}) ;
else
  Lz = 1;
end

function [matUs,loadFactorsMat] = uniaxialCompression2mat(Lx,Ly,Lz,E1,nu1,E2,nu2,p)
  %md## ONSAS path stored in an ENV variable
  addpath( genpath( getenv("ONSAS_PATH") ) );
  %md
  %md### MEBI parameters
  %md
  %md#### materials
  %md material 1:
  lambda_1 = E1*nu1/((1+nu1)*(1-2*nu1)) ; mu_1 = E1/(2*(1+nu1)) ;
  bulk_1 = E1 / ( 3*(1-2*nu1) ) ;
  materials(1).hyperElasModel = 'NHC' ;
  materials(1).hyperElasParams = [ mu_1 bulk_1 ] ;
  %md material 2:
  lambda_2 = E2*nu2/((1+nu2)*(1-2*nu2)) ; mu_2 = E2/(2*(1+nu2)) ;
  bulk_2 = E2 / ( 3*(1-2*nu2) ) ;
  materials(2).hyperElasModel = 'NHC' ;
  materials(2).hyperElasParams = [ mu_2 bulk_2 ] ;
  %md
  %md#### elements
  elements(1).elemType = 'triangle' ;
  elements(2).elemType = 'tetrahedron' ;
  elements(2).elemTypeParams = [ 2 ] ;
  %md
  %md#### boundaryConds
  %md Load:
  boundaryConds(1).loadsCoordSys = 'global';
  boundaryConds(1).loadsTimeFact = @(t) p*t ;
  boundaryConds(1).loadsBaseVals = [ 0 0 1 0 1 0 ] ;
  %md Clamped:
  boundaryConds(2).imposDispDofs = [1 2 3 4 5 6 ] ;
  boundaryConds(2).imposDispVals = [0 0 0 0 0 0 ] ;
  %
  %md
  %md#### initialConds
  initialConds = struct();
  %md
  %md### Mesh
  %md The node coordinates matrix is given by the following
  mesh.nodesCoords = [ 0    0    0 ; ...
                      0     0   Lz ; ...
                      0     Ly  Lz ; ...
                      0     Ly  0  ; ...
                      Lx/2  0   0  ; ...
                      Lx/2  0   Lz ; ...
                      Lx/2  Ly  Lz ; ...
                      Lx/2  Ly  0  ; ...
                      Lx    0   0  ; ...
                      Lx    0   Lz ; ...
                      Lx    Ly  Lz ; ...
                      Lx    Ly  0 ] ;
  %md and the connectivity cell is defined as follows:
  mesh.conecCell = {[ 0 1 1    9  12 10   ]; ... % loaded face
                    [ 0 1 1    10 12 11    ]; ... % loaded face
                    [ 0 1 2    4  1  2     ]; ... % x=0 supp face
                    [ 0 1 2    4  2  3     ]; ... % x=0 supp face
                    [ 1 2 0    1  4  2  6  ]; ... % tetrahedron
                    [ 1 2 0    6  2  3  4  ]; ... % tetrahedron
                    [ 1 2 0    4  3  6  7  ]; ... % tetrahedron
                    [ 1 2 0    4  1  5  6  ]; ... % tetrahedron
                    [ 1 2 0    4  6  5  8  ]; ... % tetrahedron
                    [ 1 2 0    4  7  6  8  ]; ... % tetrahedron
                    [ 2 2 0    5  8  6  10 ]; ... % tetrahedron
                    [ 2 2 0    10 6  7  8  ]; ... % tetrahedron
                    [ 2 2 0    8  7  10 11 ]; ... % tetrahedron
                    [ 2 2 0    8  5  9  10 ]; ... % tetrahedron
                    [ 2 2 0    8  10 9  12 ]; ... % tetrahedron
                    [ 2 2 0    8  11 10 12 ] ...  % tetrahedron
                  } ;
  %md
  %md### Analysis parameters
  %md
  analysisSettings.methodName    = 'newtonRaphson' ;
  analysisSettings.stopTolIts    = 30     ;
  analysisSettings.stopTolDeltau = 1.0e-8 ;
  analysisSettings.stopTolForces = 1.0e-8 ;
  analysisSettings.finalTime      = 1      ;
  analysisSettings.deltaT        = .01     ;
  %md
  %md### Output parameters
  otherParams.plots_format = 'vtk' ;
  otherParams.problemName = 'cantileverSolid_HandMadeMesh' ;
  %md
  [matUs, loadFactorsMat] = ONSAS( materials, elements, boundaryConds, initialConds, mesh, analysisSettings, otherParams ) ;
  %md
end

% Run ONSAS
[matUs,loadFactorsMat] = uniaxialCompression2mat(Lx,Ly,Lz,E1,nu1,E2,nu2,p) ;

% Extract dipslacements at the loaded face (x = Lx)
loadedFaceNodesIndexes = [9:12]
% dofs
dofsLoadedFaceUx = (loadedFaceNodesIndexes - 1) * 6 + 1 ;
dofsLoadedFaceUy = (loadedFaceNodesIndexes - 1) * 6 + 3 ;
dofsLoadedFaceUz = (loadedFaceNodesIndexes - 1) * 6 + 5 ;
% displacements
dispUx = matUs(dofsLoadedFaceUx, :) ;
dispUy = matUs(dofsLoadedFaceUy, :) ;
dispUz = matUs(dofsLoadedFaceUz, :) ;
% displacements at point G
dispUx_G = sum(dispUx, 1)/4
dispUy_G = sum(dispUy, 1)/4
dispUz_G = sum(dispUz, 1)/4
% plot
plotBool = true
if plotBool
  lw = 2.0 ; ms = 11 ; plotfontsize = 18 ;
  figure, hold on, grid on
  plot( p*loadFactorsMat, dispUx_G, 'r-' , 'linewidth', lw,'markersize',ms )
  plot( p*loadFactorsMat, dispUy_G, 'b-' , 'linewidth', lw,'markersize',ms )
  plot( p*loadFactorsMat, dispUz_G, 'g-' , 'linewidth', lw,'markersize',ms )
  labx = xlabel('Pressure [Pa]');   laby = ylabel('Displacement [m]') ;
  legend('u_x ', 'u_y ', 'u_z ', 'location', 'NorthWest' )
  set(gca, 'linewidth', 1.0, 'fontsize', plotfontsize )
  set(labx, 'FontSize', plotfontsize); set(laby, 'FontSize', plotfontsize) ;
  % print("./output/validation.png")
end
% print in a .txt
fid = fopen('output.txt', 'w');
fprintf(fid, '%f\n', dispUx_G(end));
fprintf(fid, '%f\n', dispUy_G(end));
fprintf(fid, '%f\n', dispUz_G(end));
fclose(fid);
