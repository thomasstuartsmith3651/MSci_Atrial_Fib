filename = "testAF1_2024_10_23_10_48_32.mat";
signal = 'epcath_uni_filt';
sinterval = 1/2034.5;
pinterval = 1/101.7250;
outputfile = "testAF1data.mat";
obj = LoadDataMatLab(filename, signal, pinterval, sinterval, outputfile);
X = obj.XCoordArray;
Y = obj.YCoordArray;
Z = obj.ZCoordArray;
PT = obj.PosTimeArray;
S = obj.SignalArray;
ST = obj.SignalTimeArray;
pF = obj.PositionSamplingFreq;
sF = obj.SignalSamplingFreq;

obj.saveDataExcel()
