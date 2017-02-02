% problem 4
clear all;
W1 = [-1 -1; -1 1; 1 0];
b1 = [1; -1; -1];
W2 = [-1 -1 -1];
b2 = 1;
a0 = [0 -1]';
n1 = W1 * a0 + b1;
a1 = poslin(W1 * a0 + b1);
a2 = W2 * a1 + b2;

s2 = -4;
F1n1 = [hardlim(n1(1)) 0 0; 0 hardlim(n1(2)) 0; 0 0 hardlim(n1(3))];
s1 = F1n1 * W2' * s2;

W21 = W2 - 0.1 * s2 * a1';
b21 = b2 - 0.1 * s2;


