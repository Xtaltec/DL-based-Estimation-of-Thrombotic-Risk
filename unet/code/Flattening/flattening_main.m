%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                   %%%%
%%%%   Code to flatten the LAA geometries from the Excel and .mat      %%%%
%%%%            data to the bullseye representation                    %%%%
%%%%                      Code By Cesar Acebes                         %%%%
%%%%                  cesaracebespinilla@gmail.com                     %%%%
%%%%                                                                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%path_base = 'C:\Users\Xabier\PhD\Frontiers\fcn\code\Flattening';
code_path ='D:\PhD\Frontiers\unet\code\Flattening';
data_path = 'D:\PhD\Frontiers\unet\data\Assemble';

%%
cd(code_path)
%% Flattening parameters %%

radial_lines=128; % From ostium towards the tip 
angular_lines=128; % 0 to 360 in each isoline
puntocentral=[-0.02,-0.01,-0.0250]; %Appproximation to the position of the circumflex
puntocentral=[-0.015,-0.015,-0.02]; %Appproximation to the position of the circumflex
t_input=1.0000e-8; %Heat equation parameter?

%% Initialization %%
addpath('./Export_fig()')
path_in = [data_path,'\Excel'];

matrix_coordinates=zeros(angular_lines*radial_lines*3);
matrix_ECAP=zeros(angular_lines*radial_lines);

% Get all files
X_all = dir(([path_in,'\X*.mat']));
F_all = dir(([path_in,'\F*.mat']));
ECAP_all = dir(([path_in,'\ECAP*.csv']));

% To store the flatten representation of each coordinate and ECAP 
mapX = zeros([length(X_all),radial_lines,angular_lines]);
mapY = zeros([length(X_all),radial_lines,angular_lines]);
mapZ = zeros([length(X_all),radial_lines,angular_lines]);
mapECAP = zeros([length(X_all),radial_lines,angular_lines]);

%% Loop in around all cases %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:length(X_all)
    
    % Loads the coordinate and connectivity data for each case
    X_struct = load([path_in,'\',X_all(i).name]);
    F_struct = load([path_in,'\',F_all(i).name]);
    Ecap_tabla=readtable([path_in,'\',ECAP_all(i).name]);
    Ecap_values=Ecap_tabla{:,:};
    X=X_struct.X_python';
    F=double(F_struct.F_python)';
    
    %%% Run the heat equation to get the flattening mapping
    [color,node_mapping, angulos_radio,color_mapping,vector_nodes,vector_coordinates]=flattening_code(X,F,radial_lines,puntocentral,angular_lines,t_input);
    
    %%% IMPORTANTE: vector_coordinates es un vector de 3*radial_lines*angular_lines x 1
    % que guarda las coordenadas x, y, z de los nodos de este modo:
    % [x radio1 angulo1;y radio1 angulo1;z radio1 angulo1;x radio1 angulo2...]
    % es decir, primero completa todo un radio, despues, un angulo y despues, x,y,z

    %asignamos valores de ECAP para hacer el bull's eye plot
    
    for j=1:size(node_mapping,1)
        for k=1:size(node_mapping,2)
            color_mapping(j,k)=Ecap_values(node_mapping(j,k));
            %%%MAPPING_FINAL_COLOR=round(MAPPING_FINAL_COLOR,0);
        end
    end
    
    %GAUSSIAN FILTERING!
    %MAPPING_FINAL_COLOR=imgaussfilt(MAPPING_FINAL_COLOR,1,'FilterSize',3,'padding','circular');

    % Asignamos valores de ECAP al vector de nodos
    vector_ECAP=reshape(color_mapping,radial_lines*angular_lines,1);
    
    %%%IMPORTANTE: vector_ECAP es un vector de radial_lines*angular_lines x 1
    % que guarda los valores de ECAP de los nodos de este modo:
    % [ECAP de radio1 angulo1;ECAP de radio1 angulo2;ECAP de radio1 angulo3...]
    % es decir, primero completa todo un radio, despues, un angulo y despues, x,y,z
    
    % Data_ECAP_Final(:,numero_auriculas)=vector_ECAP;
    
    matrix_ECAP(:,i)=vector_ECAP(:);
    matrix_coordinates(:,i)=vector_coordinates(:);
    mapeo_color_transpuesta=color_mapping';
    
    node_mapping_x=zeros(1,size(node_mapping,1)*size(node_mapping,2));
    node_mapping_y=zeros(1,size(node_mapping,1)*size(node_mapping,2));
    node_mapping_z=zeros(1,size(node_mapping,1)*size(node_mapping,2));
    num=1;
    
    for num1=1:size(node_mapping,1)
        for num2=1:size(node_mapping,2)
            node_mapping_x(num)=X(1,node_mapping(num1,num2));
            node_mapping_y(num)=X(2,node_mapping(num1,num2));
            node_mapping_z(num)=X(3,node_mapping(num1,num2));
            num=num+1;
        end
    end
    
    mapX(i,:,:) = reshape(node_mapping_x',[radial_lines,angular_lines]);
    mapY(i,:,:) = reshape(node_mapping_y',[radial_lines,angular_lines]);
    mapZ(i,:,:) = reshape(node_mapping_z',[radial_lines,angular_lines]);
    
    mapECAP(i,:,:) = mapeo_color_transpuesta; 
    disp(i)
end    

mapShape = cat(4,mapX,mapY,mapZ);

save([data_path,'/Final/Image/Shape.mat'],'mapShape')
save([data_path,'/Final/Image/ECAP.mat'],'mapECAP')

%% Save all flattenings of the LAA to images for the neural network %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Show one of the cases to ensure everything went ok
for i= 1:1
    
    figure(i)
    input = mapShape(i,:,:,3);
    imshow(reshape(input,[radial_lines,angular_lines])')
    colormap jet
    caxis([min(min(input)),max(max(input))])
    pause(1)
    close(i)
    
end

% mapX_gray = mat2gray(mapX);
% mapY_gray = mat2gray(mapY);
% mapZ_gray = mat2gray(mapZ);

% Join all 3 coordinates in a single matrix to preserve the relation
% between the 3 coordinates when changing to geyscale

Shape = zeros([size(mapX,1),3,size(mapX,2)^2]);
ShapeI = zeros([size(mapX,1),3,size(mapX,2),size(mapX,2)]);

Shape(:,1,:)= reshape( mapX,[size(mapX,1),size(mapX,2)^2]);
Shape(:,2,:)= reshape( mapY,[size(mapX,1),size(mapX,2)^2]);
Shape(:,3,:)= reshape( mapZ,[size(mapX,1),size(mapX,2)^2]);

ShapeI(:,1,:,:)= mapX;
ShapeI(:,2,:,:)= mapY;
ShapeI(:,3,:,:)= mapZ;

% Map the values to greyscale to save them as images

mapECAP_gray = mat2gray(mapECAP);
Shape_gray = mat2gray(Shape);
ShapeI_gray = mat2gray(ShapeI);

%% Plot examples
% y=1;
% figure()
% scatter3(Shape(y,1,:),Shape(y,2,:),Shape(y,3,:),30,mapECAP_gray(y,:))
% 
% figure()
% scatter3(Shape_gray(y,1,:),Shape_gray(y,2,:),Shape_gray(y,3,:),30,tempZ(y,:))

%% Save all bullseyes to png %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mkdir([data_path,'/PNG'])

for i=1:length(X_all)

    figure()
    colormap jet;
    caxis([0 1])
    IMAGEN=bullseye(reshape(ShapeI_gray(i,1,:,:),[radial_lines,angular_lines]), 'rho',[0,10],'tht0',270);
    set(gcf,'color','k');
    set(gcf, 'Position', [100 100 500 500]);
    export_fig([data_path,'/PNG/X',num2str(i,'%03.f'),'.png']);
    pause(0.1) % the pause is necessary to avoid issues when saving the png
    close all
%     saveas(IMAGEN,['./PNG/X',num2str(i),'.png'])
    
    figure()
    colormap gray;
    caxis([0 1])
    IMAGEN=bullseye(reshape(ShapeI_gray(i,2,:,:),[radial_lines,angular_lines]), 'rho',[0,10],'tht0',270);
    set(gcf,'color','k');
    set(gcf, 'Position', [100 100 500 500]);
    export_fig([data_path,'/PNG/Y',num2str(i,'%03.f'),'.png']);
    pause(0.1)
    close all
    
    figure()
    colormap gray;
    caxis([0 1])
    IMAGEN=bullseye(reshape(ShapeI_gray(i,3,:,:),[radial_lines,angular_lines]), 'rho',[0,10],'tht0',270);
    set(gcf,'color','k');
    set(gcf, 'Position', [100 100 500 500]);
    export_fig([data_path,'/PNG/Z',num2str(i,'%03.f'),'.png']);
    pause(0.1)
    close all
    
    figure()
    colormap jet;
    caxis([0 1])
    IMAGEN=bullseye(reshape(ShapeI_gray(i,1,:,:),[radial_lines,angular_lines]), 'rho',[0,10],'tht0',270);
    %set(gcf,'color','k');
    set(gcf, 'Position', [100 100 500 500]);
    caxis([min(min(ShapeI_gray(i,1,:,:))),max(max(ShapeI_gray(i,1,:,:)))])
    export_fig([data_path,'/PNG/ECAP',num2str(i,'%03.f'),'.png']);
    pause(0.1)
    close all
end

%% Choose the desired image resolution and transform all of them %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

npix=128; % Number of pixels in new images

%% Get the path to all saved PNGs

all_X = dir([data_path,'/PNG/X*.png']);
all_Y = dir([data_path,'/PNG/Y*.png']);
all_Z = dir([data_path,'/PNG/Z*.png']);
all_ECAP = dir([data_path,'/PNG/ECAP*.png']);

X = zeros(length(all_X),npix,npix);
Y = zeros(length(all_X),npix,npix);
Z = zeros(length(all_X),npix,npix);
ECAP = zeros(length(all_X),npix,npix);

% Transform all images to desired resolution

for i=1:length(all_X)
    
    png = imread([data_path,'/PNG/',all_X(i).name]);
    X(i,:,:) = imresize(png,[npix,npix]);
    
    png = imread([data_path,'/PNG/',all_Y(i).name]);
    Y(i,:,:) = imresize(png,[npix,npix]);
    
    png = imread([data_path,'/PNG/',all_Z(i).name]);
    Z(i,:,:) = imresize(png,[npix,npix]);
    
    png = imread([data_path,'/PNG/',all_ECAP(i).name]);
    ECAP(i,:,:) = imresize(png,[npix,npix]);
    %imshow (reshape(ECAP(i,:,:),[npix,npix]))
    
end

%% Save the final dataset for the DL training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save([data_path,'/Final/Bullseye/X.mat'],'X')
save([data_path,'/Final/Bullseye/Y.mat'],'Y')
save([data_path,'/Final/Bullseye/Z.mat'],'Z')
save([data_path,'/Final/Bullseye/ECAP.mat'],'ECAP')

%%
% 
figure()
axis('tight');
colormap Jet(256);
lighting none
set(gcf,'color','w');
imagesc(mapeo_color_transpuesta)
xlabel('Radial isolines')
ylabel('Samples of each isoline')
caxis([0 5])
title(['LAA number ' num2str(1)])
%
%
% % % % % % este codigo es para imprimir la estructura 3d coloreada
nueva=reshape(mapeo_color_transpuesta,1,radial_lines*angular_lines);

%% Este codigo es para representar en 3d los nodos obtenidos con el flattening

node_mapping_x=zeros(1,size(node_mapping,1)*size(node_mapping,2));
node_mapping_y=zeros(1,size(node_mapping,1)*size(node_mapping,2));
node_mapping_z=zeros(1,size(node_mapping,1)*size(node_mapping,2));
num=1;

for num1=1:size(node_mapping,1)
    for num2=1:size(node_mapping,2)
        node_mapping_x(num)=X(1,node_mapping(num1,num2));
        node_mapping_y(num)=X(2,node_mapping(num1,num2));
        node_mapping_z(num)=X(3,node_mapping(num1,num2));
        num=num+1;
    end
end

figure()
axis('tight');
colormap Jet(256);
lighting none
set(gcf,'color','w');
scatter3(node_mapping_x,node_mapping_y,node_mapping_z,50,nueva,'filled')
caxis([0 5])
