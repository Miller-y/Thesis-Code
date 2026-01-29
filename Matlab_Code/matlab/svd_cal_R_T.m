%% 使用SVD计算旋转和平移
% 论文 Least-Squares Fitting of Two 3-D Point Sets
function [R,T,W] = svd_cal_R_T(p11,p12,p13,p21,p22,p23)
% p11,p12,p13 分别表示传感器计算出来的坐标
% p21,p22,p23 分别表示led在世界坐标系的坐标
p1 = (p11+p12+p13)/3
p2 = (p21+p22+p23)/3
p11_ = p11-p1;
p12_ = p12-p1;
p13_ = p13-p1;
p21_ = p21-p2;
p22_ = p22-p2;
p23_ = p23-p2;


W = p21_*p11_';
W = W + p22_*p12_';
W = W + p23_*p13_'
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

