function L = dlt_calib_L(leds,Bias)

X = leds(1,:);
Y = leds(2,:);
Z = leds(3,:);
sample = 1:length(leds);%[1,6,7,10,11,13,16];
size  = length(sample);
L = zeros(7,length(Bias(:,1)));
for cn = 1:length(Bias(:,1))
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
end

