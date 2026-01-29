%% 通过L系数计算

H =zeros(8,5);
Z = zeros(8,1);
cnt = 1;
LLL = L;
for i = 1:2:8
    H(i,1) = LLL(1:3,cnt)'* LLL(5:7,cnt);
    H(i,2) = LLL(5:7,cnt)'* LLL(5:7,cnt);
    H(i,3:5) = 0;
    H(i+1,1:2) = 0;
    H(i+1,3) = LLL(1:3,cnt)'* LLL(1:3,cnt);
    H(i+1,4) = LLL(5:7,cnt)'* LLL(5:7,cnt);
    H(i+1,5) = 2*LLL(1:3,cnt)'* LLL(5:7,cnt);
    Z(i) = 0;
    Z(i+1) = LLL(5:7,cnt)'* LLL(5:7,cnt)*2500*2500;
    cnt  = cnt + 1;  
end

X = (H'*H)\(H'*Z)


H =zeros(length(LLL(1,:)),1);
Z = zeros(length(LLL(1,:)),1);
cnt = 1;
for i = 1:1:length(LLL(1,:))
    H(i,1) = LLL(5:7,cnt)'* LLL(5:7,cnt);
    Z(i) = LLL(1:3,cnt)'* LLL(5:7,cnt);;  
    cnt  = cnt + 1;
end

X = (H'*H)\(H'*Z)

lam = X;
H =zeros(length(LLL(1,:)),1);
Z = zeros(length(LLL(1,:)),1);
cnt = 1;
for i = 1:1:length(LLL(1,:))
    h11 =  LLL(1:3,cnt)'* LLL(1:3,cnt);
    h22 = LLL(5:7,cnt)'* LLL(5:7,cnt);
    h12 = LLL(1:3,cnt)'* LLL(5:7,cnt);
    H(i,1) = h11+h22*lam*lam - 2*h12*lam;
    Z(i) = h22;
    cnt  = cnt + 1;
end

X = (H'*H)\(H'*Z);
sqrt(1/X)