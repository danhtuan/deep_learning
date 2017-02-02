% prob 3
clear all;
G = [-1 0 1; 1 1 1]';
t = [-1.5 0.5 2.5]';
c = t' * t;
d = -2 * G' * t;
A = 2 * G' * G; 
x = [0 0]';
x1 = x - 0.1 * (A * x + d);
f0 = 0.5 * x' * A * x + d' * x + c;
f1 = 0.5 * x1' * A * x1 + d' * x1 + c;


