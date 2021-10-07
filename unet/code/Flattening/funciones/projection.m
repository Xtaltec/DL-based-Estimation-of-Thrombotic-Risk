
function [px,py,pz]=projection(a,b,c,d,x,y,z)
% given an plane equation ax+by+cz=d, project points xyz onto the plane
% return the coordinates of the new projected points
% written by Neo Jing Ci, 11/7/18
A=[1 0 0 -a; 0 1 0 -b; 0 0 1 -c; a b c 0];
B=[x; y; z; d];
X=A\B;
px=X(1);
py=X(2);
pz=X(3);
end
