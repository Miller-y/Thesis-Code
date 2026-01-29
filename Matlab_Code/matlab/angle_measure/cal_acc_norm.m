%%
function g = cal_acc_norm(acc)

for i = 1:length(acc)
   g(i) = norm(acc(i,:));
end
end

