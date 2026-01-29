function [lambda0,f] = cal_lambda0_f(LLL)
% LLL： 单个ccd对应的多个L系数(7*n 的大小)

H =zeros(length(LLL(1,:)),1);
Z = zeros(length(LLL(1,:)),1);
cnt = 1;
for i = 1:1:length(LLL(1,:))
    H(i,1) = LLL(5:7,cnt)'* LLL(5:7,cnt);
    Z(i) = LLL(1:3,cnt)'* LLL(5:7,cnt); 
    cnt  = cnt + 1;
end
lambda0 = (H'*H)\(H'*Z);

H =zeros(length(LLL(1,:)),1);
Z = zeros(length(LLL(1,:)),1);
cnt = 1;
for i = 1:1:length(LLL(1,:))
    h11 =  LLL(1:3,cnt)'* LLL(1:3,cnt);
    h22 = LLL(5:7,cnt)'* LLL(5:7,cnt);
    h12 = LLL(1:3,cnt)'* LLL(5:7,cnt);
    H(i,1) = h11+h22*lambda0*lambda0 - 2*h12*lambda0;
    Z(i) = h22;
    cnt  = cnt + 1;
end

X = (H'*H)\(H'*Z);
f = sqrt(1/X);
end

