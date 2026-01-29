%% 根据led的数据计算L系数
% 11.25数据 ccd少了16 led少了13 只用1-12个数据

% [X Y Z 1 -λX -λY -λZ][L1 L2 L3 L4 L5 L6 L7]' = λ
% mean_led = mean(leds);
% mean_led(3) = 0;
% leds = leds - mean_led;
% for i = 1:length(leds)
%    leds(i,:) = leds(i,:)*rotx(30);
% end
noise = 0;
X = leds(1,:)+noise*(rand(1,length(leds))-0.5); % led 世界坐标系坐标的X
Y = leds(2,:)+noise*(rand(1,length(leds))-0.5); % led 世界坐标系坐标的Y
Z = leds(3,:)+noise*(rand(1,length(leds))-0.5); % led 世界坐标系坐标的Z
sample = 1:length(leds);%[1,6,7,10,11,13,16];
% sample = 1:7;
size  = length(sample);
L = [];
for cn = 1:3
    A = zeros(size,7);
    B = zeros(size,1);
    for i = 1:size
        A(i,1) = X(sample(i));
        A(i,2) = Y(sample(i));
        A(i,3) = Z(sample(i));
        A(i,4) = 1;
        
        A(i,5) = -(Bias(cn,sample(i)))*X(sample(i));
        A(i,6) = -(Bias(cn,sample(i)))*Y(sample(i));
        A(i,7) = -(Bias(cn,sample(i)))*Z(sample(i));
        B(i) = Bias(cn,sample(i));
    end
%     B = Bias(cn,sample)'+2;
    if size == 7
        L(:,cn) = A\B;
    else
        L(:,cn) = (A'*A)\(A'*B);
    end
end
%% 
% %% 1
% A = zeros(size,7);
% for i = 1:size
%    A(i,1) = X(i);
%    A(i,2) = Y(i);
%    A(i,3) = Z(i);
%    A(i,4) = 1;
%    A(i,5) = -bias1(i)*X(i);
%    A(i,6) = -bias1(i)*Y(i);
%    A(i,7) = -bias1(i)*Z(i); 
% end
% B = bias1(1:size)';
% 
% % L1 = (A'*B)\(A'*A)
% % L1 = inv(A'*A)*(A'*B)
% L1 = (A'*A)\(A'*B)
% % L1 = inv(A)*B
% %% 2
% for i = 1:size
%    A(i,1) = X(i);
%    A(i,2) = Y(i);
%    A(i,3) = Z(i);
%    A(i,4) = 1;
%    A(i,5) = -bias2(i)*X(i);
%    A(i,6) = -bias2(i)*Y(i);
%    A(i,7) = -bias2(i)*Z(i); 
% end
% B = bias2(1:size)';
% 
% % L1 = (A'*B)\(A'*A)
% % L1 = inv(A'*A)*(A'*B)
% L2 = (A'*A)\(A'*B)
% % L1 = inv(A)*B
% 
% %% 3
% for i = 1:size
%    A(i,1) = X(i);
%    A(i,2) = Y(i);
%    A(i,3) = Z(i);
%    A(i,4) = 1;
%    A(i,5) = -bias3(i)*X(i);
%    A(i,6) = -bias3(i)*Y(i);
%    A(i,7) = -bias3(i)*Z(i); 
% end
% B = bias2(1:size)';
% 
% % L1 = (A'*B)\(A'*A)
% % L1 = inv(A'*A)*(A'*B)
% L3 = (A'*A)\(A'*B)
% % L1 = inv(A)*B

%% 根据L系数计算计算参数
%% 1
LL = L(:,1);
z1 = 1/norm(LL(5:7));
r3 = LL(5:7)*z1;
lambda1 = (LL(1:3)*z1)'*r3;
f1 = norm(((LL(1:3)*z1)-lambda1*r3));
r1 = ((LL(1:3)*z1)-lambda1*r3)/f1;
r2 = cross(r3,r1);
R1 = [r1';r2';r3';];
x1 = (LL(4)-lambda1)*z1/f1;
%% 2
LL = L(:,2);
z2 = 1/norm(LL(5:7));
r3 = LL(5:7)*z2;
lambda2 = (LL(1:3)*z2)'*r3
f2 = norm(((LL(1:3)*z2)-lambda2*r3));
r1 = ((LL(1:3)*z2)-lambda2*r3)/f2;
r2 = cross(r3,r1);
R2 = [r1';r2';r3';];
x2 = (LL(4)-lambda2)*z2/f2
%% 3
LL = L(:,3);
z3 = 1/norm(LL(5:7));
r3 = LL(5:7)*z3;
lambda3 = (LL(1:3)*z3)'*r3
f3 = norm(((LL(1:3)*z3)-lambda3*r3));
r1 = ((LL(1:3)*z3)-lambda3*r3)/f3;
r2 = cross(r3,r1);
R3 = [r1';r2';r3';];
x3 = (LL(4)-lambda2)*z3/f3;

f1 = [f1,f2,f3];
z = [z1,z2,z3];
x = [x1,x2,x3];
lambda = [lambda1,lambda2,lambda3];
