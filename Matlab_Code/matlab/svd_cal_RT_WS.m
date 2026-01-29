%% 使用SVD计算旋转和平移
% 论文 Least-Squares Fitting of Two 3-D Point Sets
function [R,T] = svd_cal_RT_WS(M_W,M_S)
p11 = M_S(:,1);p12 = M_S(:,2);p13 = M_S(:,3);
p21 = M_W(:,1);p22 = M_W(:,2);p23 = M_W(:,3);
p1 = (p11+p12+p13)/3;
p2 = (p21+p22+p23)/3;
p11_ = p11-p1;
p12_ = p12-p1;
p13_ = p13-p1;
p21_ = p21-p2;
p22_ = p22-p2;
p23_ = p23-p2;


W = p21_*p11_';
W = W + p22_*p12_';
W = W + p23_*p13_';
[U,S,V] = svd(W);
R = U*V';
if(det(R) < 0)
    % NOTE: 不是R = -R
    V(:,3) = - V(:,3);
    %     R = V*U';
    R = U*V';
end
T = p2 - R*p1;
end