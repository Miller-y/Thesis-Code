
function g = cal_acc_var(a,N)
g = ones(1,length(a));
va = ones(1,N)*10;

for i = 1:length(g)
    va(end+1) = a(i);
    va(1) = [];
    g(i) = var(va);
end

end

