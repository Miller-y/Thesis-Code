alpha  = 10;
gama = 3;
beta = 0;
g1_g2 = [];
for i = 0:10:90
    beta = i;
    r_beta = rotx(beta);
    r_g_a = rotx(gama)*rotz(alpha);
    
    r_imu = r_beta*r_g_a;
    
    rv = r_g_a'*[0,1,0]';
    Rv = rotationVectorToMatrix(30/180*pi*rv);
    
    g1 = r_imu'*[0,0,1]';
    g2 = Rv*g1;
    g1_g2(end+1) = norm(g1 - g2);
end

Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 5];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(20);
Fig.Name = '1';
% Fig.Color = 'w';
% 创建axes对象, 设定坐标轴属性
clear ax;
ax = axes();
ax.Parent = Fig;
ax.XGrid = 'off';
ax.YGrid = 'off';
ax.GridLineStyle = '-.';
ax.Visible = 'on';
ax.Box = 'on';
% ax.Position = [0.1344,0.13,0.806,0.82];
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(20);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.XLim = [-180 180];
% ax.YLim = [-0.02 0.16];
ax.XLabel.String = '$\beta$ /$^{\circ}$';
ax.YLabel.String = '$||\bf{g}_1 - \bf{g}_0||$';
ax.XLabel.Interpreter = 'latex';
ax.YLabel.Interpreter = 'latex';
plot(0:10:90,g1_g2,'--o','linewidth',1);

filename = '..\figure\beta_g1_g0';
print(filename,'-dpng','-r600');
filename = '..\..\latex\figures\beta_g1_g0';
print(filename,'-depsc','-r600');

%%
function [ss1,ss2] = cal_ss(g1,g2,g3)
ss1 = g2 -g1;
ss2 = g3 - g2;
end

function a = cal_alpha(ss1,ss2)
a = atand((ss1(2)*ss2(3)-ss2(2)*ss1(3))/(-ss1(1)*ss2(3) + ss2(1)*ss1(3)));
end

function ga = cal_gama(ss1,a)
ga = atand((ss1(1)*sind(a)+ss1(2)*cosd(a))/ss1(3));
end

function b = cal_beta(g,a,ga)
b = asind((g(1)*sind(a)+g(2)*cosd(a))*cosd(ga) - g(3)*sind(ga));
end

function angle  = cal_angle(g,a,b,ga, flag)
x = g(1);
y = g(2);
z = g(3);
ca = cosd(a);
sa = sind(a);
cb = cosd(b);
sb = sind(b);
cga = cosd(ga);
sga = sind(ga);

angle = acosd( (z+sga*sb) / (cga*cb));
if(flag)
    %if(~((45<=angle&&angle<135) || (-45>=angle&&angle>=-135)))
    angle = -asind((x*ca-y*sa)/cb);
end
end

function g = cal_axis(a1,a2,a3)

a_err(1,:) = a2-a1;
a_err(2,:) = a3-a2;
g = cross(a_err(1,:),a_err(2,:))/norm(cross(a_err(1,:),a_err(2,:)));
if(g(1) < 0)
    g = -g;
end
end

function na = add_white_noise(noise_power, len )
fs=100;
na = sqrt(fs/2*(noise_power))*randn(len,1);
end

