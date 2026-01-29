list = [];
row = 1;
column = 1;
c_list = {};
g = cal_acc_norm(acc);
var_N = cal_acc_var(g,10);
count = 1;
last_index = 1;
isZupt = false;
% acc_up = 9.86;
% acc_bottom = 9.66;
acc_up = 10.04;
acc_bottom = 9.96;
var_limit = 0.001;
duration_in = 10;
duration_out = 3;
for i = 1:length(g)
    if isZupt == false
        if (g(i)<=acc_up) && (g(i)>=acc_bottom) && (var_N(i)<var_limit)
            if(count >= duration_in)
                isZupt = true;
                count = duration_out;
                c_list{end+1} = [];
            else
                % continue
                if(i-last_index) == 1
                    count = count+1;
                else % 不连续 清
                    count = 1;
                end
                last_index = i;
            end
        end
    end
    
    % 进入 零速
    if isZupt==true
        if (g(i)>acc_up) || (g(i)<acc_bottom)||(var_N(i)>=var_limit)
            if(count <= 1)
                isZupt = false;
            else
                % continue
                if(i-last_index) == 1
                    count = count-1;
                else % 不连续 清
                    count = duration_out;
                end
                last_index = i;
            end
        end
    end
    
    if(isZupt == true)
        list(end+1)=i;
        c_list{end}(end+1)=i;
    end
end


