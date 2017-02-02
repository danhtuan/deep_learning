% problem 1
clear all;
close all;
W1 = [-1 1]';
b1 = [0.5 1]';
W2 = [1 1];
b2 = -1;

p = -10:0.1:10;
a2 = poslin(-p + 0.5) + poslin(p + 1) - 1;
figure;
plot(p, a2);
