function [color,MAPPING_FINAL_NODOS, angulos_radio,MAPPING_FINAL_COLOR,vector_nodes,vector_coordinates] = flattening_LAA_Acebes_mejoradoo(X,F,radial_lines,puntocentral,angular_lines,t_input)
% Modelo hecho a partir de puntos en todo el ostium, con lineas
% equicaloricas  
% Los inputs son: 
%   La malla en formato .off
%   El numero de lineas paralelas al ostium que queremos (numberoflines)

%   Las coordenadas de la circunfleja (puntocentral)
% Esta funcion te devuelve:
%   Los puntos de cada linea paralela al ostium mas cercano a la
%   circunfleja 
%   A partir de cada uno de estos, los puntos de cada linea ortogonal


% load('D:\PhD\Rasmus\DL\Code_cesar\Flattening\Excel\X_python1.mat')%es X
% load('D:\PhD\Rasmus\DL\Code_cesar\Flattening\Excel\F_python1.mat')%es F
% 
% load('D:\PhD\DL\Data\CNN\Excel\X_python102.mat')%es X
% load('D:\PhD\DL\Data\CNN\Excel\F_python102.mat')%es F

% X= X_python'/1000;
% F = double(F_python');
% 
% angular_lines=100;
% radial_lines=100;
% puntocentral=[-0.01,0.05,-0.05];
% t_input=0.00001;

%con este codigo ploteamos los nodos del LAA
%  figure()
%  scatter3(X(1,:),X(2,:),X(3,:),2,'k','filled')
%  hold on;
%  scatter3(puntocentral(1),puntocentral(2),puntocentral(3),10,'r','filled')

getd = @(p)path(p,path);
getd('funciones');

clear options.face_vertex_color;

n = size(X,2);
m = size(F,2);
options.lighting = 1;
axis('tight');

XF = @(i)X(:,F(i,:)+1);%he cambiado lo de +1
%XF = @(i)X(:,F(i,:));
Na = cross( XF(2)-XF(1), XF(3)-XF(1) );
amplitude = @(X)sqrt( sum( X.^2 ) );
A = amplitude(Na)/2;
normalize = @(X)X ./ repmat(amplitude(X), [3 1]);
N = normalize(Na);
I = []; J = []; V = []; % indexes to build the sparse matrices
for i=1:3
    % opposite edge e_i indexes
    s = mod(i,3)+1;
    t = mod(i+1,3)+1;
    % vector N_f^e_i
    wi = cross(XF(t)-XF(s),N);
    % update the index listing
    I = [I, 1:m];
    J = [J, F(i,:)+1];
    V = [V, wi];
end
dA = spdiags(1./(2*A(:)),0,m,m);
GradMat = {};
for k=1:3
    GradMat{k} = dA*sparse(I,J,V(k,:),m,n);
end
Grad = @(u)[GradMat{1}*u, GradMat{2}*u, GradMat{3}*u]';
dAf = spdiags(2*A(:),0,m,m);
DivMat = {GradMat{1}'*dAf, GradMat{2}'*dAf, GradMat{3}'*dAf};
Div = @(q)DivMat{1}*q(1,:)' + DivMat{2}*q(2,:)' + DivMat{3}*q(3,:)';
Delta = DivMat{1}*GradMat{1} + DivMat{2}*GradMat{2} + DivMat{3}*GradMat{3};
cota = @(a,b)cot( acos( dot(normalize(a),normalize(b)) ) );
I = []; J = []; V = []; % indexes to build the sparse matrices
Ia = []; Va = []; % area of vertices
for i=1:3
    s = mod(i,3)+1;
    t = mod(i+1,3)+1;
    % adjacent edge
    ctheta = cota(XF(s)-XF(i), XF(t)-XF(i));
    % ctheta = max(ctheta, 1e-2); % avoid degeneracy
    % update the index listing
    I = [I, F(s,:)+1, F(t,:)+1];
    J = [J, F(t,:)+1, F(s,:)+1];
    V = [V, ctheta, ctheta];
    % update the diagonal with area of face around vertices
    Ia = [Ia, F(i,:)+1];
    Va = [Va, A];
end
Ac = sparse(Ia,Ia,Va,n,n);
Wc = sparse(I,J,V,n,n);
DeltaCot = spdiags(full(sum(Wc))', 0, n,n) - Wc;
f = X(2,:);
options.face_vertex_color = f(:);
g = Delta*f(:);
g = clamp(g, -3*std(g), 3*std(g));
options.face_vertex_color = rescale(g);
%Heat difussion and time stepping
delta = zeros(n,1);
%ahora voy a seleccionar todos los nodos del borde
bordes=compute_boundary(F+1);


% %con el siguiente codigo se plotea el punto inicial y el final (prueba)
% % punto_inicial=punto_cercano_en_linea(bordes,puntocentral,X)
% % punto_final=punto_lejano_en_linea(bordes,puntocentral,X)
% %Asi, lo que hago, es poner solo 2 fuentes, una en el punto mas cercano a
% %la circunfleja y una en el punto mas lejano a ella
% % % % % % % % % % % % % % delta(punto_inicial)=1;
% % % % % % % % % % % % % % delta(punto_final)=1;






%bordes_coordenadas=zeros(3,length(bordes));
%bordes coordenadas es una matriz que guarda las coordenadas de los
%nodos del borde (es decir, aquellos que detecta en el borde)

for num=1:length(bordes)
    bordes_coordenadas(:,num)=X(:,bordes(num));
end
bordes_coordenadas_final = ordenar_angulos(bordes_coordenadas,bordes,puntocentral);



% % % %hacemos que cada nodo del borde tenga el valor de 1
for nodo_borde=1:size(bordes,2)
    delta(bordes(nodo_borde))=1;
end



%DEFINIMOS t EN EL INPUT
%t_input = 1;

u = (Ac+t_input*DeltaCot)\delta;

g = Grad(u);
h = -normalize(g);
%
% h(h==Inf)=1;
% h(h==-Inf)=0;
%
phi1 = Delta \ Div(h);
phi_inverted=-phi1;
%lo hemos invertido para hacer que en el borde este el valor minimo de
%calor

%a partir de aqui, vamos a coger todos los puntos pertenecientes a la
%isoterma de nuestro interes
phi_normalized_pre=(phi_inverted-min(phi_inverted));
phi_normalized=phi_normalized_pre/max(phi_normalized_pre);
%aqui hemos hecho que phi este entre 0 y 1, esta normalizado
Matriz_lineas=zeros(radial_lines);
%threshold=-0.995;%para valores mas bajos, se selecciona la slice
threshold=-0.998;%para valores mas bajos, se selecciona la slice%%%%%%%

phi_final=phi_normalized;
contador=ones(1,radial_lines);

%%%%%%%%%representamos todos los nodos con sus distancias (+ alto=+ cerca del
%%%%%%ostium)
% scatter3(X(1,:),X(2,:),X(3,:),5,phi_final)
% colorbar
% hold on;



for aa=1:length(bordes)
    Matriz_lineas(radial_lines,aa)=bordes(aa);
    phi_final(bordes(aa))=0;
end

for aaa=1:n
    %esta es una prueba
    for bbb=1:radial_lines-1
        %hago esto para hacer que la ultima linea sean solo los bordes
        if -cos(2*pi*phi_normalized(aaa)-2*pi*bbb/radial_lines)<threshold
            phi_final(aaa)=0;%esto es para representarlo
            Matriz_lineas(bbb,contador(bbb))=aaa;
            contador(bbb)=contador(bbb)+1;
            break
        end
    end
end
for aaa=1:n
    if phi_final(aaa)~=0%esto es para representarlo
        phi_final(aaa)=1;
    end
end
%hasta aqui, Matriz_lineas tiene en cada fila los nodos de cada isoterma




LINEAS_ORDENADAS=struct();

%ahora vamos a obtener el plano que mejor se ajuste al set de puntos que 
% pertenecen a la isoterma
matriz_planos=zeros(3,radial_lines);
points=zeros(3,radial_lines);
% % % % % % % % % % figure()
% % t=[0:0.1:0.1*radial_lines];
% % numberOfFrames = radial_lines;
% % % Set up the movie structure.
% % % Preallocate recalledMovie, which will be an array of structures.
% % % First get a cell array with all the frames.
% % allTheFrames = cell(numberOfFrames,1);
% % vidHeight = 344;
% % vidWidth = 446;
% % allTheFrames(:) = {zeros(vidHeight, vidWidth, 3, 'uint8')};
% % % Next get a cell array with all the colormaps.
% % allTheColorMaps = cell(numberOfFrames,1);
% % allTheColorMaps(:) = {zeros(256, 3)};
% % % Now combine these to make the array of structures.
% % myMovie = struct('cdata', allTheFrames, 'colormap', allTheColorMaps);
% % % Create a VideoWriter object to write the video out to a new, different file.
% % % writerObj = VideoWriter('problem_3.avi');
% % % open(writerObj);
% % % Need to change from the default renderer to zbuffer to get it to work right.
% % % openGL doesn't work and Painters is way too slow.
% % set(gcf, 'renderer', 'zbuffer');

for bbb=1:radial_lines
    coordenadas_linea=zeros(3,1);
    linea=Matriz_lineas(radial_lines-bbb+1,:);
    linea( :, ~any(linea,1) ) = [];
    %asi eliminamos las columnas con ceros y nos quedan solo los nodos que
    %nos interesan
    for ccc=1:length(linea)
        coordenadas_linea(:,ccc)=X(:,linea(ccc));
    end
    %coordenadas_linea contiene las coordenadas de cada nodo de esa linea
    
    %a partir de ahora codigo de:
    %https://www.mathworks.com/matlabcentral/answers/448708-plane-fitting-a-3d-scatter-plot
    x=coordenadas_linea(1,:);
    y=coordenadas_linea(2,:);
    z=coordenadas_linea(3,:);
    
    [normal,basis,point] = affine_fit(coordenadas_linea');
    B = [x(:) y(:) ones(size(x(:)))] \ z(:);
    xv = linspace(min(x), max(x), 10)';
    yv = linspace(min(y), max(y), 10)';
    [X_grid,Y_grid] = meshgrid(xv, yv);
    Z = reshape([X_grid(:), Y_grid(:), ones(size(X_grid(:)))] * B, numel(xv), []);

%%%%%%%%el siguiente codigo sirve para representar 3 cosas:
    
   %%%estas lineas representan todos los nodos del LAA
%     scatter3(X(1,:),X(2,:),X(3,:),1,'k','filled')
%     hold on
%     %%estas lineas representan los nodos de las isolines obtenidas del heat method
%     scatter3(x,y,z, 15,'filled')
%     hold on
%     set(gcf,'color','w');

% % % %     view(120, 60)
% % % %     thisFrame = getframe(gca);
    %%%esta linea representa el plano correspondiente a cada isolinea
%     mesh(X_grid, Y_grid, Z, 'FaceAlpha', 0.5)
%     hold on
%     axis('tight');
%     colormap Jet(256);
%     lighting none
%     set(gcf,'color','w');
%     view(-120, 35)

    %title(sprintf('Z = %+.3E\\cdotX %+.3E\\cdotY %+.3E', B))
    
    matriz_planos(:,bbb)=normal;%
    points(:,bbb) = point;
end
%matriz_planos es una matriz que contiene los parametros del plano que
%mejor se ajusta al set de puntos de cada isoterma.


%%%%%%%%%%%%%%%%%%%%%%
%%%%vamos a hacer una prueba
%%%%%%%%%%%%%%%%%%%%%%
%circunfleja de cada 
puntocentral=puntocentral';
linea=Matriz_lineas(radial_lines,:);%hago que linea sean los nodos de la isolinea
linea( :, ~any(linea,1) ) = [];%elimino los ceros
punto_slice=puntocentral;%inicializo punto_slice
distancia_max=-inf;%inicializo la distancia minima como infinito
for ccc=1:length(linea)
    %linea(ccc) es el nodo que nos interesa
    %X(linea(ccc)) son las coordenadas del nodo que nos interesa
    Coordenadas_del_nodo=X(:,linea(ccc));
    distancia=norm(punto_slice-Coordenadas_del_nodo);
    if distancia > distancia_max
        distancia_max=distancia;
        nodo_con_distancia_max=linea(ccc);
        puntocentral=Coordenadas_del_nodo;%nuevo
    end
    %asi hemos calculado el punto mas cercano a:
    %a la circunfleja, si estamos en la slice del ostium.
    %al punto mas cercano a la circunfleja de la slice anterior, si no
    %estamos en el ostium.
end
%nodo_con_distancia_min es el nodo que nos interesa
%%%nodo_con_distancia_min es el nodo del ostium + cercano a la circunfleja
% scatter3(X(1,:),X(2,:),X(3,:),1,'k','filled')
% hold on;
% scatter3(X(1,nodo_con_distancia_max),X(2,nodo_con_distancia_max),X(3,nodo_con_distancia_max),10,'b','filled')

delta = zeros(n,1);
delta(nodo_con_distancia_max)=1;

u = (Ac+t_input*DeltaCot)\delta;
g = Grad(u);
h = -normalize(g);
phi = Delta \ Div(h);
%%%%cuanto mayor phi, mas distancia desde el punto del ostium + cercano a
%%%%la circunfleja, como se puede ver en la siguiente imagen
% figure()
% scatter3(X(1,:),X(2,:),X(3,:),10,phi,'filled')
% colormap('jet')
% colorbar
nodos_cercanos_a_circunfleja=zeros(radial_lines,1);
nodos_cercanos_a_circunfleja(radial_lines)=nodo_con_distancia_max;

for bbb=1:radial_lines-1
    bbb_inverso=radial_lines-bbb+1;%hago esto para empezar desde el ostium
    linea=Matriz_lineas(bbb_inverso,:);%hago que linea sean los nodos de la isolinea
    linea( :, ~any(linea,1) ) = [];%elimino los 0s
    distancia_max=-Inf;%inicializo la distancia maxima como 0
    for ccc=1:length(linea)
        %linea(ccc) es el nodo que nos interesa
        %X(linea(ccc)) son las coordenadas del nodo que nos interesa
        Distancia_del_nodo=phi(linea(ccc));
        if Distancia_del_nodo > distancia_max
            distancia_max=Distancia_del_nodo;
            nodo_con_distancia_max=linea(ccc);
        end
        %asi hemos calculado el punto mas cercano a:
        %a la circunfleja, si estamos en la slice del ostium.
        %al punto mas cercano a la circunfleja de la slice anterior, si no
        %estamos en el ostium.
    end
    %nodo_con_distancia_min es el nodo que nos interesa
    nodos_cercanos_a_circunfleja(bbb)=nodo_con_distancia_max;
%%%asi se pueden ver los puntos en los que comienza el mapeo angular
%     scatter3(X(1,:),X(2,:),X(3,:),1,'k')
%     hold on;
%     scatter3(X(1,nodo_con_distancia_max),X(2,nodo_con_distancia_max),X(3,nodo_con_distancia_max),10,'b','filled')
%     hold on;
end




coordenadas_proyeccion_=zeros(3,1,radial_lines);
coordenadas_proyeccion=zeros(3,1,radial_lines);

%es una matriz que contiene las coordenadas de las lineas proyectadas

%obtenemos la recta que va desde el punto + alejado del ostium al
%centro del ostium
[min_phi,nodo_mas_lejano]=min(phi_normalized);
ostium=Matriz_lineas(radial_lines,:);
ostium( :, ~any(ostium,1) ) = [];%obtenemos los nodos del ostium
%scatter3(X(1,nodo_mas_lejano),X(2,nodo_mas_lejano),X(3,nodo_mas_lejano),20,'k')
ostium_x=mean(X(1,ostium(:)));
ostium_y=mean(X(2,ostium(:)));
ostium_z=mean(X(3,ostium(:)));
dist_x=X(1,nodo_mas_lejano)-ostium_x;
dist_y=X(2,nodo_mas_lejano)-ostium_y;
dist_z=X(3,nodo_mas_lejano)-ostium_z;
vector_ostium_lejos=[dist_x;dist_y;dist_z];
%%%nodo_mas_lejano es el nodo mas alejado del ostium
%%%en la siguiente linea se puede ver graficado el vector que va del
%%%centro del ostium al nodo mas alejado
%quiver3(ostium_x,ostium_y,ostium_z,2*dist_x,2*dist_y,2*dist_z,'k')

% % % % % % % % % % figure()
for bbb=1:radial_lines
    linea=Matriz_lineas(bbb,:);
    linea( :, ~any(linea,1) ) = [];
    
    circ_x=puntocentral(1);
    circ_y=puntocentral(2);
    circ_z=puntocentral(3);
    
    a = matriz_planos(1,bbb);
    b = matriz_planos(2,bbb);
    c = matriz_planos(3,bbb);
    d = matriz_planos(:,bbb)'*points(:,bbb);
    
%     a=matriz_planos(1,bbb);
%     b=matriz_planos(2,bbb);
%     c=-1;
%     d=matriz_planos(3,bbb);
    
    [p_circ_x,p_circ_y,p_circ_z]=projection(a,b,c,d,circ_x,circ_y,circ_z);
    for ccc=1:length(linea)

        x=X(1,linea(ccc));
        y=X(2,linea(ccc));
        z=X(3,linea(ccc));

        [px,py,pz]=projection(a,b,c,d,x,y,z);
        coordenadas_proyeccion_(1,ccc,bbb)=px;
        coordenadas_proyeccion_(2,ccc,bbb)=py;
        coordenadas_proyeccion_(3,ccc,bbb)=pz;
        
    end
    
    
    %para proyectarlo en un plano
    centro_x=mean(X(1,linea(:)));
    centro_y=mean(X(2,linea(:)));
    centro_z=mean(X(3,linea(:)));
    
    
    centro_px=sum(coordenadas_proyeccion_(1,:,bbb))/nnz(coordenadas_proyeccion_(1,:,bbb));
    centro_py=sum(coordenadas_proyeccion_(2,:,bbb))/nnz(coordenadas_proyeccion_(2,:,bbb));
    centro_pz=sum(coordenadas_proyeccion_(3,:,bbb))/nnz(coordenadas_proyeccion_(3,:,bbb));
    
    distx=centro_x-centro_px;
    disty=centro_y-centro_py;
    distz=centro_z-centro_pz;

    
    
    p_circ_x=p_circ_x+distx;
    p_circ_y=p_circ_y+disty;
    p_circ_z=p_circ_z+distz;
    %[p_circ_x,p_circ_y,p_circ_z] es la proyeccion de la circunfleja
    
    
    %
    %inicializamos variables
    x_=[];
    y_=[];
    z_=[];
    nodos_proyectados=[];
    contador=1;

    for ccc=1:length(linea)
        coordenadas_proyeccion(1,ccc,bbb)=coordenadas_proyeccion_(1,ccc,bbb)+distx;
        coordenadas_proyeccion(2,ccc,bbb)=coordenadas_proyeccion_(2,ccc,bbb)+disty;
        coordenadas_proyeccion(3,ccc,bbb)=coordenadas_proyeccion_(3,ccc,bbb)+distz;
        
        
        %el siguiente bloque de codigo sirve para eliminar los elementos
        %proyectados en [0;0;0]
        if coordenadas_proyeccion(1,ccc,bbb) ~= 0
            x_(contador)=coordenadas_proyeccion(1,ccc,bbb);
            if coordenadas_proyeccion(2,ccc,bbb) ~= 0
                y_(contador)=coordenadas_proyeccion(2,ccc,bbb);
                if coordenadas_proyeccion(3,ccc,bbb) ~= 0
                    z_(contador)=coordenadas_proyeccion(3,ccc,bbb);
                    nodos_proyectados(contador)=linea(ccc);
                    %todos los nodos que hemos proyectado estaran aqui
                    
                    contador=contador+1;
                end
            end
        end
        
    end
    
%%%%%El siguiente codigo sirve para representar:
   
%    %%%la proyeccion de la circunfleja
%    scatter3(p_circ_x,p_circ_y,p_circ_z,10,'b','filled')%ploteamos la proyeccion de la circunfleja
%    hold on;

%%%%%%aqui se pueden ver la proyeccion de los nodos de cada isolinea en su plano
%     scatter3(X(1,:),X(2,:),X(3,:),1,'k','filled')
%     hold on
%     scatter3(x_,y_,z_, 15,'filled')
%     hold on
%      set(gcf,'color','w');
%     %%los nodos de cada isoline mas cercanos a la circunfleja
%     abab=X(:,nodos_cercanos_a_circunfleja(bbb));
%     scatter3(abab(1),abab(2),abab(3),'k', 'filled')
%     axis('tight');
%     colormap Jet(256);
%     lighting none
%     set(gcf,'color','w');
%     hold on
%     title(sprintf('Z = %+.3E\\cdotX %+.3E\\cdotY %+.3E', B))    
    
    

    
    
    %ahora vamos a ordenar los nodos por el angulo que forman con la
    %circunfleja
    proyeccion_circunfleja=[p_circ_x;p_circ_y;p_circ_z];
    centro_proyeccion=[centro_x;centro_y;centro_z];
    %nodos_proyectados lo hemos creado antes
    %puntocentral es un input de la funcion 
    %primero, vamos a calcular los angulos 
    for num=1:length(nodos_proyectados)
        %creamos dos vectores que salen del centro, uno hacia la
        %circunfleja y el otro hacia el nodo que nos interesa
        %asi se comprueba si es positivo o negativo
        %source: https://www.mathworks.com/matlabcentral/answers/476800-loosing-quadrant-data-using-atan2-when-finding-angle-between-3-points-in-3d-space-in-matlab
        
        %creo que esta linea soluciona el problema de la circunfleja (creo)
        %proyeccion_circunfleja=X(:,nodos_cercanos_a_circunfleja(bbb));%%%esto fallaba!!!!!
%         scatter3(proyeccion_circunfleja(1),proyeccion_circunfleja(2),proyeccion_circunfleja(3))
%         hold on         
        vector1=proyeccion_circunfleja-centro_proyeccion;
        vector2=coordenadas_proyeccion(1:3,num,bbb)-centro_proyeccion;
        d=2*vector_ostium_lejos;
        %d = [0;0;1]; % your preference direction
        x = cross(vector1,vector2);
        nx = norm(x);
        A = atan2d(nx, dot(vector1,vector2));
        x = x / nx;
        if dot(x,d) < 0
            x = -x;
            A = 360-A;
        end
        coordenadas_proyeccion(4,num,bbb)=A;
        

        
        %scatter3(coordenadas_proyeccion(1,num,1),coordenadas_proyeccion(2,num,1),coordenadas_proyeccion(3,num,1),coordenadas_proyeccion(4,num,1),'filled')
        %hold on;
%         
        %asi guardamos el numero de nodo
        coordenadas_proyeccion(5,num,bbb)=nodos_proyectados(num);
    end
    
    %asi transponemos la matriz para ordenarla de acuerdo a sus filas (lo
    %que antes eran columnas)
    coordenadas_proyeccion_tr=coordenadas_proyeccion(:,:,bbb)';
    
    %lo ordenamos
    coordenadas_proyeccion_tr_ordenado=sortrows(coordenadas_proyeccion_tr,4);
    
    %deshacemos la transposicion
    coordenadas_proyeccion_ordenado=coordenadas_proyeccion_tr_ordenado';
    coordenadas_proyeccion_ordenado=removeMatZeros(coordenadas_proyeccion_ordenado);
    
    
    %we create a cell for every slice
    LINEAS_ORDENADAS.("slice"+bbb)=coordenadas_proyeccion_ordenado;
    
    %el output de esta matrix es:
    %en las tres primeras filas, las coordenadas de los nodos
    %en la cuarta fila el angulo que forma ese nodo con la orejuela
    %en la quinta fila, el nodo
    
%     x_ordenada=coordenadas_proyeccion_ordenado(1,:);
%     y_ordenada=coordenadas_proyeccion_ordenado(2,:);
%     z_ordenada=coordenadas_proyeccion_ordenado(3,:);
%     
%     
%     nodes=scatter3(x_ordenada,y_ordenada,z_ordenada, 'filled');
%     axis('tight');
%     colormap Jet(256);
%     lighting none
%     set(gcf,'color','w');
%     title(num2str(bbb))
%     color=(bordes_coordenadas_final(4,:)/bordes_coordenadas_final(4,end));
%     nodes.MarkerFaceColor = [color color color];
%     view(-120, 35)
%     hold on
    
%     for ccc=1:size(bordes_coordenadas_final,2)
%         nodes=scatter3(x_ordenada(ccc),y_ordenada(ccc),z_ordenada(ccc), 'filled');
%         axis('tight');
%         colormap Jet(256);
%         lighting none
%         set(gcf,'color','w');
%         color=(bordes_coordenadas_final(4,ccc)/bordes_coordenadas_final(4,end));
%         nodes.MarkerFaceColor = [color color color];
%         view(-120, 35)
%     end
    
    
    
    
    
    
%     xlim([25 75])
%     ylim([60 135])
%     zlim([-190 -150])
%     hold on
    %title(sprintf('Z = %+.3E\\cdotX %+.3E\\cdotY %+.3E', B))
    
end
close all
% 
% p = 0.2;%ponia 30 pero lo he cambiao
% DispFunc = @(phi_normalized)(phi_normalized);
% options.face_vertex_color = DispFunc(phi_normalized);
% color=options.face_vertex_color;
% figure()
% clf;
% plot_mesh(X,F,options);
% figure()
% plot_mesh(X,F,options);
% axis('tight');
% colormap Jet(256);
% lighting none
% set(gcf,'color','w');

%radial_lines es el numero de puntos que se plotearan en cada slice (el
%numero de radios diferentes)
MAPPING_FINAL=zeros(radial_lines,angular_lines);
MAPPING_FINAL_NODOS=zeros(radial_lines,angular_lines);
MAPPING_FINAL_ANGULOS=zeros(radial_lines,angular_lines);
MAPPING_FINAL_COLOR=zeros(radial_lines,angular_lines);
vector_nodes=zeros(radial_lines*angular_lines,1);
for bbb=1:radial_lines
    angulos_radio=linspace(0,360,angular_lines+1);
    angulos_radio=angulos_radio(1:end-1);
    for ccc=1:length(angulos_radio)
        %try
        estructura=LINEAS_ORDENADAS.("slice"+bbb);
        matriz_valor=angulos_radio(ccc)*ones(1,size(estructura,2));
        %asi se selecciona el angulo con valor mas cercano al que queremos
        [minValue,closestIndex]=min(abs(estructura(4,:)-matriz_valor));
        %IMPORTANTE:SOLO LO HE HECHO TENIENDO EN CUENTA EL NODO CON UN
        %VALOR MAS CERCANO, NO HE HECHO INTERPOLACIONES!!!!!!!!!
        %en estas dos matrices, cada fila es una slice (polar-based
        %mapping) y cada columna una "columna" del radial mapping:
        %LINEAS_ORDENADAS
        if size(estructura(4,closestIndex),2) ~= 0
            MAPPING_FINAL_ANGULOS(bbb,ccc)=estructura(4,closestIndex);
            %en esta matriz, se pueden ver los nodos de los que se han de tomar
            %valores para plotearse
            MAPPING_FINAL_NODOS(bbb,ccc)=estructura(5,closestIndex);
            %aqui guardamos el color de cada nodo para representarlo
            nodo=estructura(5,closestIndex);
            MAPPING_FINAL_COLOR(bbb,ccc)=phi_normalized(nodo);
            estructurabien=estructura;%guardo una estructura que sé que funciona
        else
            estructura=estructurabien;
            matriz_valor=angulos_radio(ccc)*ones(1,size(estructura,2));
            %asi se selecciona el angulo con valor mas cercano al que queremos
            [minValue,closestIndex]=min(abs(estructura(4,:)-matriz_valor));
            MAPPING_FINAL_ANGULOS(bbb,ccc)=estructura(4,closestIndex);
            %en esta matriz, se pueden ver los nodos de los que se han de tomar
            %valores para plotearse
            MAPPING_FINAL_NODOS(bbb,ccc)=estructura(5,closestIndex);
            %aqui guardamos el color de cada nodo para representarlo
            nodo=estructura(5,closestIndex);
            MAPPING_FINAL_COLOR(bbb,ccc)=phi_normalized(nodo);
            %este if else lo he puesto porque a veces hay slices vacias, en
            %ese caso, asi se salta a la siguiente slice
        end
    end
end

%I perform a reshaping in order to present the nodes that I export in form
%of a vertical vector (Nx1), where N is the total number of nodes
vector_nodes=reshape(MAPPING_FINAL_NODOS,radial_lines*angular_lines,1);


%now, I will save the geometric data in the vector_coordinates matrix

vector_coordinates=zeros(length(vector_nodes)*3,1);

for contador=0:length(vector_nodes)-1
    %we save the coordinate X:
    vector_coordinates(contador*3+1,1)=X(1,vector_nodes(contador+1));
    %we save the coordinate Y:
    vector_coordinates(contador*3+2,1)=X(2,vector_nodes(contador+1));
    %we save the coordinate Z:
    vector_coordinates(contador*3+3,1)=X(3,vector_nodes(contador+1));
end

color=phi_normalized;

%toc