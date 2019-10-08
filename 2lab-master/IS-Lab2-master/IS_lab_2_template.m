%Classification using perceptron
clc
clear
%Reading apple images
x=0:1/19:1;
T=(1+0.6*sin(2*pi*x/0.7)+0.3*sin(2*pi*x))/2;
figure(1)
plot(x,T,'kx');
%%

%% train multi perceptron with one inputs and one output

%first layer
% generate random initial values of w1 and b
w11 = randn(1);
w12 = randn(1);
w13 = randn(1);
w14 = randn(1);
b11 = randn(1);
b12 = randn(1);
b13 = randn(1);
b14 = randn(1);


v11=zeros(length(x),1);
v12=zeros(length(x),1);
v13=zeros(length(x),1);
v14=zeros(length(x),1);
y11=zeros(length(x),1);
y12=zeros(length(x),1);
y13=zeros(length(x),1);
y14=zeros(length(x),1);
% calculate wieghted sum with randomly generated parameters
for i = 1:20
    v11(i)=x(i)*w11+b11;
    v12(i)=x(i)*w12+b12;
    v13(i)=x(i)*w13+b13;
    v14(i)=x(i)*w14+b14;
    % calculate current output of the perceptron 
    y11(i)=1/(1+exp(-v11(i)));
    y12(i)=1/(1+exp(-v12(i)));
    y13(i)=1/(1+exp(-v13(i)));
    y14(i)=1/(1+exp(-v14(i)));
end

%second layer
w21 = randn(1);
w22 = randn(1);
w23 = randn(1);
w24 = randn(1);
b21 = randn(1);
y_2 = zeros(length(x),1);
e1=zeros(length(x),1);

for i = 1:20
    y_2(i)=y11(i)*w21+y12(i)*w22+y13(i)*w23+y14(i)*w24+b21;
    e1(i)=T(i)-y_2(i);
end



% calculate the total error for these inputs 
e=0;
e_abs=0;
for i = 1:20
    e=e+e1(i)^2/2;
end

%training_step
n=0.02;
%max error
error=0.001;
% write training algorithm
j=0;

figure(1)
hold on
plot(x,T,'kx');
plot(x,y_2,'ro');
hold off


while (e>error) % executes while the total error is not ((e_abs > error)||0
    j=j+1;
	% here should be your code of parameter update
%   calculate output for current example
% 
%   calculate error for current example
% 
%   update parameters of output layer
    for i= 1:20
        w21 = w21 + n*e1(i)*y11(i);
        w22 = w22 + n*e1(i)*y12(i);
        w23 = w23 + n*e1(i)*y13(i);
        w24 = w24 + n*e1(i)*y14(i);
        b21 = b21 + n*e1(i);
        
        w11 = w11 + n*e1(i)*w21*x(i)*exp(b11+w11*x(i))/((exp(b11+w11*x(i))+1)^2);
        w12 = w12 + n*e1(i)*w22*x(i)*exp(b12+w12*x(i))/((exp(b12+w12*x(i))+1)^2);
        w13 = w13 + n*e1(i)*w23*x(i)*exp(b13+w13*x(i))/((exp(b13+w13*x(i))+1)^2);
        w14 = w14 + n*e1(i)*w24*x(i)*exp(b14+w14*x(i))/((exp(b14+w14*x(i))+1)^2);
        
        b11 = b11 + n*e1(i)*w21*exp(b11+w11*x(i))/((exp(b11+w11*x(i))+1)^2);
        b12 = b12 + n*e1(i)*w22*exp(b12+w12*x(i))/((exp(b12+w12*x(i))+1)^2);
        b13 = b13 + n*e1(i)*w23*exp(b13+w13*x(i))/((exp(b13+w13*x(i))+1)^2);
        b14 = b14 + n*e1(i)*w24*exp(b14+w14*x(i))/((exp(b14+w14*x(i))+1)^2);
        
       
    end
% 
%   Test how good are updated parameters (weights) on all examples used for training
%   calculate outputs and errors for all 5 examples using current values of the parameter set {w1, w2, b}
    for i = 1:20
        v11(i)=x(i)*w11+b11;
        v12(i)=x(i)*w12+b12;
        v13(i)=x(i)*w13+b13;
        v14(i)=x(i)*w14+b14;
        % calculate current output of the perceptron 
        y11(i)=1/(1+exp(-v11(i)));
        y12(i)=1/(1+exp(-v12(i)));
        y13(i)=1/(1+exp(-v13(i)));
        y14(i)=1/(1+exp(-v14(i)));
    end
    
    for i = 1:20
        y_2(i)=y11(i)*w21+y12(i)*w22+y13(i)*w23+y14(i)*w24+b21;
        e1(i)=T(i)-y_2(i);
    end
    e=0;
    e_abs=0;
    for i = 1:20
        e=e+e1(i)^2/2;
    end
end
figure(2)
hold on
plot(x,T,'kx');
plot(x,y_2,'ro');
hold off
% figure(1)
% x=0:0.01:1;
% f=-b/w2-x*w1/w2;
% plot(x,f,'g')
% hold on
% plot(hsv_value_A1,metric_A1,'b*')
% plot(hsv_value_A2,metric_A2,'b*')
% plot(hsv_value_A3,metric_A3,'b*')
% plot(hsv_value_A4,metric_A4,'b*')
% plot(hsv_value_A5,metric_A5,'b*')
% plot(hsv_value_A6,metric_A6,'b*')
% plot(hsv_value_A7,metric_A7,'b*')
% plot(hsv_value_A8,metric_A8,'b*')
% plot(hsv_value_A9,metric_A9,'b*')
% plot(hsv_value_P1,metric_P1,'ro')
% plot(hsv_value_P2,metric_P2,'ro')
% plot(hsv_value_P3,metric_P3,'ro')
% plot(hsv_value_P4,metric_P4,'ro')
% hold off
% 
% v=hsv_value_A4*w1+metric_A4*w2+b;
% figure(2)
% subplot(4,2,1)
% image(A4);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_A5*w1+metric_A5*w2+b;
% figure(2)
% subplot(4,2,2)
% image(A5);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_A6*w1+metric_A6*w2+b;
% figure(2)
% subplot(4,2,3)
% image(A6);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
%     
% v=hsv_value_A7*w1+metric_A7*w2+b;
% figure(2)
% subplot(4,2,4)
% image(A7);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_A8*w1+metric_A8*w2+b;
% figure(2)
% subplot(4,2,5)
% image(A8);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_A9*w1+metric_A9*w2+b;
% figure(2)
% subplot(4,2,6)
% image(A9);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_P3*w1+metric_P3*w2+b;
% figure(2)
% subplot(4,2,7)
% image(P3);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end
% 
% v=hsv_value_P4*w1+metric_P4*w2+b;
% figure(2)
% subplot(4,2,8)
% image(P4);
% if v>0
%     title('obuolys');
% else
%     title('kriause');
% end