%% 计算旋转轴

acc_mean = [];
% figure;
% hold on;
for i = 1:length(c_list)
    % 静止时间大于1秒
    if(length( c_list{i}) >1000)
        acc_mean(end+1,:) = mean(acc(c_list{i},:));
%         plot(c_list{i},acc(c_list{i},1))
    end
end

axis_r(1,:)=cal_axis(acc_mean(1,:),acc_mean(2,:),acc_mean(3,:));
axis_r(2,:)=cal_axis(acc_mean(2,:),acc_mean(4,:),acc_mean(10,:));
axis_r(3,:)=cal_axis(acc_mean(1,:),acc_mean(4,:),acc_mean(6,:));
axis_r = [];
for i =1:(length(acc_mean)-1)
    for j =(i+1):(length(acc_mean)-1)
        for k =(j+1):(length(acc_mean)-1)
            a = cal_axis(acc_mean(i,:),acc_mean(j,:),acc_mean(k,:));
            if(a(1)<0)
                a = -a;
            end
            axis_r = [axis_r;a];
        end
    end
end

%%
function g = cal_axis(a1,a2,a3)

a_err(1,:) = a2-a1; 
a_err(2,:) = a3-a2;

g = cross(a_err(1,:),a_err(2,:))/norm(cross(a_err(1,:),a_err(2,:)));
end