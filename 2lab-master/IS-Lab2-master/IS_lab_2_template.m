%Classification using perceptron
clc
clear
%sudaromi pradiniai taskai
x=0:1/19:1;
T=(1+0.6*sin(2*pi*x/0.7)+0.3*sin(2*pi*x))/2;
figure(1)
plot(x,T,'kx');

%% train multi perceptron with one inputs and one output

%first layer
% generate random initial values of w1x and bx
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
    %paskaiciuojamas rezultatas ir palyginamas su norimu gauti, taip
    %isskaiciuojama klaida
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
    
%   update parameters
    for i= 1:20
        %keiciami parametrai antrame sluoksnyje, imant isvestine pagal
        %keiciama parametra nuo galutinio rezultato
        w21 = w21 + n*e1(i)*y11(i);
        w22 = w22 + n*e1(i)*y12(i);
        w23 = w23 + n*e1(i)*y13(i);
        w24 = w24 + n*e1(i)*y14(i);
        b21 = b21 + n*e1(i);
        
        %keiciami parametrai pirmame sluoksnyje, imant isvestine pagal
        %keiciama parametra nuo galutinio rezultato. cia isvestine
        %sudetingesne, nes pirmame sluoksnyje naudojama netiesine funkcija
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
%   is naujo skaiciuojami pirmo sluoksnio isejimai
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
    %is naujo skaiciuojami antro sluoksnio isejimai
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
%palyginami gauti rezultatai su turimais gauti
figure(2)
hold on
plot(x,T,'kx');
plot(x,y_2,'ro');
hold off

%Atvaizduojama tiksli kreive gauta is perceptrono
x=0:1/199:1;
 for i = 1:200
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
y_3=zeros(length(x));
for i = 1:200
    y_3(i)=y11(i)*w21+y12(i)*w22+y13(i)*w23+y14(i)*w24+b21;
end

figure(3)
plot(x,y_3);
