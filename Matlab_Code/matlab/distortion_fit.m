% double xn = M_C.x() / M_C.z();
% 	double h = -0.1132 * pow(xn, 2) + 0.173438 * pow(xn, 4);
% 	sm_u_real = sm_f * xn +
% 				sm_f * h * xn +
% 				sm_u_0;
% 	u = sm_u_real;
% 	sm_u_ideal = xn * sm_f + sm_u_0;
K0 = [-0.1226,-0.02102,0.163,0.2359,0.01405];
% hh = @(k,x,y) k(1)*x.^2 + k(2)*y.^2+ k(3)*x.^2.*y.^2+k(4)*x.^4+k(5)*y.^4;
h = @(k,x) x(:,1) + x(:,1).*(k(1)*x(:,1).^2 + k(2)*x(:,2).^2+ k(3)*(x(:,1).^2).*(x(:,2).^2)+k(4)*x(:,1).^4+k(5)*x(:,2).^4);
% h = @(k) k(1)*x.^2 + k(2)*y.^2+ k(3)*x.^2.*y.^2+k(4)*x.^4+k(5)*y.^4;
xy  = [xn(:) yn(:)];
[K,resnorm,r]=lsqcurvefit(h,K0,xy,xn_(:));

kk = [-0.1134 0.2359 ];
hh = @(k,x) x + x.*(k(1)*x.^2 + k(2)*x.^4);
% h = @(k) k(1)*x.^2 + k(2)*y.^2+ k(3)*x.^2.*y.^2+k(4)*x.^4+k(5)*y.^4;
[KK,resnorm,r]=lsqcurvefit(hh,kk,xn(:),xn_(:));
% lsqnonlin
% function [outputArg1,outputArg2] = untitled(inputArg1,inputArg2)
% %UNTITLED 此处显示有关此函数的摘要
% %   此处显示详细说明
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% end
