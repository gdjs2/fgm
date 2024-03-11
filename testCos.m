adjMat1 = [0, 1, 1; 1, 0, 0; 1, 0, 0];
Pt1 = [1, 1, 1; 1, 1, 1];
par1 = st('link', 'cos', 'val', adjMat1);
g1 = newGphA(Pt1, par1);
g1

adjMat2 = [0, 1, 0; 1, 0, 1; 0, 1, 0];
Pt2 = [1, 1, 1; 1, 1, 1];
par2 = st('link', 'cos', 'val', adjMat2);
g2 = newGphA(Pt2, par2);
g2

parKnl = st('alg', 'cos');
gphs = {g1, g2};
[KP, KQ]= conKnlGphPQD(gphs, parKnl);
Ct = ones(size(KP));
asg = fgmD(KP, KQ, Ct, gphs, [], []);

asg