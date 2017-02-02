% problem 2
[p1 p2] = meshgrid(-5:0.1:5);
a2 = -poslin(- p1 - p2 + 1) - poslin(- p1 + p2 - 1) - poslin(p1 - 1) + 1;
figure;
surfc(p1, p2, a2);
hold on;
contour3(p1, p2, a2);