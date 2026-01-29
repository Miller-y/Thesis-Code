
led_ans = zeros(3,length(leds));

for cnt = 1:length(leds)
    A = zeros(3,3);
    B = zeros(3,1);
    for i = 1:3
        A(i,1) = (Bias(i,cnt))*L(5,i) - L(1,i);
        A(i,2) = (Bias(i,cnt))*L(6,i) - L(2,i);
        A(i,3) = (Bias(i,cnt))*L(7,i) - L(3,i);
        B(i) = L(4,i)-(Bias(i,cnt));
    end
    led_ans(:,cnt) = A\B;   
end
led_ans' - leds'
