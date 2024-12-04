classdef LoadDataMatLab
    properties
        %insert properties of class
        MatLabFileName
        PositionSamplingInterval
        PositionSamplingFreq
        SignalSamplingInterval
        SignalSamplingFreq
        OutputFileName
        SignalType
        PositionIndex
        SignalIndex
        PosVarPathName
        SigVarPathName
        XCoordArray
        YCoordArray
        ZCoordArray
        PosTimeArray
        SignalArray
        SignalTimeArray
    end
    
    methods
        %initialisation if the formats are consistent
        function obj = LoadDataMatLab(MatLabFileName, SignalType, PositionSamplingInterval, SignalSamplingInterval, OutputFileName)
            %defining matlab file name
            obj.MatLabFileName = load(MatLabFileName);
            %define signal type to be loaded
            obj.SignalType = SignalType;
            %finds row index of position data where the electrode name is "HD"
            obj.PositionIndex = find(strcmp({obj.MatLabFileName.userdata.locations.catheters.name}, "HD"));
            %finds tensor of "HD" electrode locations
            obj.PosVarPathName = obj.MatLabFileName.userdata.locations.catheters(obj.PositionIndex).egms;
            %finds row index of desired signal where the electrode name is "HD"
            obj.SignalIndex = find(strcmp({obj.MatLabFileName.userdata.(obj.SignalType).catheters.name}, "HD"));
            %finds 2D array of "HD" electrode signals
            obj.SigVarPathName = obj.MatLabFileName.userdata.(obj.SignalType).catheters(obj.SignalIndex).egms; %calls unipolar raw data
            %defining position and signal sampling time intervals
            obj.PositionSamplingInterval = PositionSamplingInterval;
            obj.SignalSamplingInterval = SignalSamplingInterval;
            obj.PositionSamplingFreq = obj.MatLabFileName.userdata.locations.sampleFreq;
            obj.SignalSamplingFreq = obj.MatLabFileName.userdata.(obj.SignalType).sampleFreq;
            %calculating arrays
            [obj.XCoordArray, obj.YCoordArray, obj.ZCoordArray, obj.PosTimeArray, obj.SignalArray, obj.SignalTimeArray] = readDataMatLab(obj);
            %file name for outputted .mat file
            obj.OutputFileName = OutputFileName;
        end
        
        %read var
        function [xArr, yArr, zArr, posTimeArr, sigArr, sigTimeArr] = readDataMatLab(obj)
            %signal table of each electrode: column = electrode number
            sigVar = obj.SigVarPathName;
            sigArr = sigVar(:,1:16);
            %position table of each electrode
            posArr = obj.PosVarPathName;
            %x array of each electrode: row = time, column = electrode number
            xArr = posArr(:,1:16,1);
            %y array of each electrode: row = time, column = electrode number
            yArr = posArr(:,1:16,2);
            %z array of each electrode: row = time, column = electrode number
            zArr = posArr(:,1:16,3);
            %time array for signals
            sig_end_time = (height(sigArr) - 1) * obj.SignalSamplingInterval;
            sigTimeArr = 0:obj.SignalSamplingInterval:sig_end_time;
            %time array for positions
            pos_end_time = (height(posArr) - 1) * obj.PositionSamplingInterval;
            posTimeArr = 0:obj.PositionSamplingInterval:pos_end_time;
        end

        %save array as .mat file
        function saveDataMatLab(obj)
            X = obj.XCoordArray;
            Y = obj.YCoordArray;
            Z = obj.ZCoordArray;
            S = obj.SignalArray;
            pT = obj.PosTimeArray;
            pF = obj.PositionSamplingFreq;
            sT = obj.SignalTimeArray;
            sF = obj.SignalSamplingFreq;
            save(obj.OutputFileName, "X", "Y", "Z", "S", "pT", "sT", "pF", "sF");
        end

        function saveDataExcel(obj)
            filename = 'testAF1data.xlsx';
            writematrix(obj.XCoordArray, filename, 'Sheet', 'X coordinates');
            writematrix(obj.YCoordArray, filename, 'Sheet', 'Y coordinates');
            writematrix(obj.ZCoordArray, filename, 'Sheet', 'Z coordinates');
            writematrix(obj.SignalArray, filename, 'Sheet', 'Signals');
            %writematrix(obj.PosTimeArray, filename, 'Sheet', 'Position time array');
            %writematrix(obj.SignalTimeArray, filename, 'Sheet', 'Signal time array');
            writematrix(obj.PositionSamplingFreq, filename, 'Sheet', 'Position Sampling Frequency');
            writematrix(obj.SignalSamplingFreq, filename, 'Sheet', 'Signal Sampling Frequency');
        end
    end
end