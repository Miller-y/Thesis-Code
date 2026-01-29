%% 畸变数据整理
xn = zeros(13,13);
yn = zeros(13,13);
alp = zeros(13,13);
beta = zeros(13,13);
distortion = zeros(13,13);
delt_d = zeros(13,13);
dd = PredictedY- RealY;
x_real = zeros(13,13);
x_ideal = zeros(13,13);
ff = zeros(13,13); % 焦距
for index = 1:length(YField)
    alp(i(index)+7,j(index)+7) = YField(index); % Y方向入射角，论文对应X
    beta(i(index)+7,j(index)+7) = XField(index); % X方向入射角，论文对应Y
    % 求tan
    xn(i(index)+7,j(index)+7) = tand(YField(index));
    yn(i(index)+7,j(index)+7) = tand(XField(index));
    % 实际
    x_real(i(index)+7,j(index)+7) = RealY(index);
    x_ideal(i(index)+7,j(index)+7) = PredictedY(index);
    ff(i(index)+7,j(index)+7) = x_ideal(i(index)+7,j(index)+7)/xn(i(index)+7,j(index)+7);
    % 畸变
    distortion(i(index)+7,j(index)+7) = -PredictedY(index)+ RealY(index);
    % 相对畸变
    if PredictedY(index)~=0
        delt_d(i(index)+7,j(index)+7) = (RealY(index) - PredictedY(index))/PredictedY(index);
    end
end
% s = surf(alp,beta,distortion);
% s = surf(xn,yn,delt_d);
% s.EdgeColor = 'none';

%% 计算函数g(xn_,yn_)
% 实际
xn_ = xn;
yn_ = yn;
delt_d_ = delt_d;
for i = 1:length(xn)
    for j = 1:length(xn)
        xn_(i,j) = delt_d(i,j)*xn(i,j)+xn(i,j);
        yn_(i,j) = delt_d(i,j)*yn(i,j)+yn(i,j);
        if xn_(i,j)~=0
            delt_d_(i,j) = (xn_(i,j)-xn(i,j))/xn_(i,j);
        end
    end
end
% s = surf(xn_,yn_,delt_d_);

%% 画 beta 引起的误差图
dis_y_err = zeros(1,13);
for in = 1:13
%     plot(beta(:,in),distortion(:,in))
    dis_y_err(in) = max(distortion(:,in)) - min(distortion(:,in));
%     hold on;
end
dis_y_err = dis_y_err * 1e3;

%% 画 x 的相对畸变
dis_x = delt_d(7,:);
xn_no_y = xn(7,:);
% plot(xn(7,:),dis_x);
% 实际
xn_no_y_ = dis_x.*xn(7,:)+xn(7,:);
dis_x_ = (xn_no_y_-xn_no_y)./xn_no_y_;
dis_x_(7) = 0;


%% 画参考文献中的拟合曲面
% p1 =-0.4217;
% p2 = -0.4011;
% p3 = 0.3478;
% p4 = 0.2174;
% p5 = 0.1631;
%
% a = -25:1:25;
% b = -25:1:25;
% x = tand(a);
% y = tand(b);
% [xn,yn]=meshgrid(x,y);
% [an,bn]=meshgrid(a,b);
% h = p1*xn.^2+p2*yn.^2+p3*xn.^2.*yn.^2+p4*xn.^4+p5*yn.^4;
% h = h.*xn*47;
% figure
% s = surf(an,bn,h)
% s.EdgeColor = 'none';