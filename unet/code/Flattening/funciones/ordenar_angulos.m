function bordes_coordenadas_final = ordenar_angulos(proyeccion_coordenadas,nodos_proyectados,proyeccion_circunfleja)
%esta funcion sirve para ordenar los nodos de la variable bordes en funcion
%del angulo que forman con el centro de todos los elementos de esa variable
%y la circunfleja.
%
%Los inputs son:
%proyeccion_coordenadas: una matriz (n x 3), donde n es el numero de nodos y
%que contiene las coordenadas de estos
%nodos_proyectados: una matriz (n x 1), donde n es el numero de nodos y que contiene 
%los nodos de los bordes (aunque solo lo usaremos para la longitud)
%proyeccion_circunfleja: (1 x 3)son las coordenadas de la circunfleja


%voy a calcular las coordenadas del punto central del ostium
centro_x=mean(proyeccion_coordenadas(1,:));
centro_y=mean(proyeccion_coordenadas(2,:));
%voy a calcular cada angulo (en primer lugar, aquellos entre 0 y pi rad)
bordes_angulos=zeros(1,length(nodos_proyectados));
for num=1:length(nodos_proyectados)
    x=proyeccion_coordenadas(1,num);
    y=proyeccion_coordenadas(2,num);
    if x-centro_x > 0 && y-centro_y > 0
        bordes_angulos(num)=atan((y-centro_y)/(x-centro_x));
    elseif x-centro_x < 0 && y-centro_y > 0
        bordes_angulos(num)=pi+atan((y-centro_y)/(x-centro_x));
    elseif x-centro_x < 0 && y-centro_y < 0
        bordes_angulos(num)=pi+atan((y-centro_y)/(x-centro_x));
    elseif x-centro_x > 0 && y-centro_y < 0
        bordes_angulos(num)=2*pi+atan((y-centro_y)/(x-centro_x));
    end
end
%esta linea de codigo es para ordenarlos:
[bordes_angulos_ordenados, order] = sort(bordes_angulos);
bordes_angulos_grados_ordenados=bordes_angulos_ordenados*180/pi;
%order es un vector que pone la posicion (en plan, si en la posicion 1 de
%order pone 50, es que la posicion 1
proyeccion_coordenadas(4,:)=order;
proyeccion_coordenadas(5,:)=nodos_proyectados;
proyeccion_coordenadas(6,:)=bordes_angulos;
%en las lineas de codigo de antes he puesto:
%en la cuarta fila el orden que corresponde
%en la quinta fila el numero de nodo que corresponde
%en la sexta fila, el angulo que forma ese nodo con el eje x
bordes_coordenadas_tr=proyeccion_coordenadas';
%lo he transpuesto
bordes_coordenadas_tr_ordenado=sortrows(bordes_coordenadas_tr,4);
%lo he ordenado
bordes_coordenadas_ordenado=bordes_coordenadas_tr_ordenado';
%he deshecho la transposicion y en la matriz bordes_coordenadas_ordenado
%las tres primeras filas son las coordenadas x y z, la cuarta fila es el
%numero de orden a partir del nodo mas cercano al 0,0 y la quinta el nodo
%del LAA al que corresponde

%bueno, hasta aqui los hemos ordenado desde el (0,0), pero queremos
%ordenarlos desde la circunfleja.
%para ello, lo primero que hago es calcular el angulo que forma la
%circunfleja:
circunfleja_x=proyeccion_circunfleja(1);
circunfleja_y=proyeccion_circunfleja(2);
if circunfleja_x-centro_x > 0 && circunfleja_y-centro_y > 0
    angulo_circunfleja=atan((circunfleja_y-centro_y)/(circunfleja_x-centro_x));
elseif circunfleja_x-centro_x < 0 && circunfleja_y-centro_y > 0
    angulo_circunfleja=pi+atan((circunfleja_y-centro_y)/(circunfleja_x-centro_x));
elseif circunfleja_x-centro_x < 0 && circunfleja_y-centro_y < 0
    angulo_circunfleja=pi+atan((circunfleja_y-centro_y)/(circunfleja_x-centro_x));
elseif circunfleja_x-centro_x > 0 && circunfleja_y-centro_y < 0
    angulo_circunfleja=2*pi+atan((circunfleja_y-centro_y)/(circunfleja_x-centro_x));
end

%vamos a ver cual de los nodos esta mas cerca de la circunfleja (es decir,
%hay una menor diferencia entre sus angulos
resta=inf;
for numero=1:length(nodos_proyectados)
    if abs(bordes_coordenadas_ordenado(6,numero)-angulo_circunfleja) > 0
        if bordes_coordenadas_ordenado(6,numero)-angulo_circunfleja < resta
            resta=bordes_coordenadas_ordenado(6,numero)-angulo_circunfleja;
            nodo_inicial=numero;%asi sabemos cual es por el que hay que empezar
        end
    end
end

%vamos a escribir el nuevo orden en el que han de estar de acuerdo a la
%circunfleja
for numero=1:length(nodos_proyectados)
    if bordes_coordenadas_ordenado(4,numero)<nodo_inicial
        bordes_coordenadas_ordenado(4,numero)=bordes_coordenadas_ordenado(4,numero)-nodo_inicial+length(nodos_proyectados)+1;
    elseif bordes_coordenadas_ordenado(4,numero)>nodo_inicial
        bordes_coordenadas_ordenado(4,numero)=bordes_coordenadas_ordenado(4,numero)-nodo_inicial+1;
    elseif bordes_coordenadas_ordenado(4,numero)==nodo_inicial
        bordes_coordenadas_ordenado(4,numero)=1;
    end
end
%lo ordenamos de nuevo
bordes_coordenadas_ordenado_tr_2=bordes_coordenadas_ordenado';
%lo he transpuesto
bordes_coordenadas_final_tr=sortrows(bordes_coordenadas_ordenado_tr_2,4);
%lo he ordenado
bordes_coordenadas_final=bordes_coordenadas_final_tr';

%ya he ordenado un conjunto de puntos de forma consecutiva desde la
%circunfleja 
end