classdef LoadDataMatLab
    properties
        %insert properties of class
        MatLabFileName
        PositionIndex
        SignalIndex
        PosVarPathName
        SigVarPathName
        PositionSamplingInterval
        SignalSamplingInterval
        XCoordArray
        YCoordArray
        ZCoordArray
        PosTimeArray
        SignalArray
        SignalTimeArray
    end
    
    methods
        %initialisation if the formats are consistent
        function obj = LoadDataMatLab(MatLabFileName, PositionSamplingInterval, SignalSamplingInterval)
            %defining matlab file name
            obj.MatLabFileName = load(MatLabFileName);
            %finds row index of position data where the electrode name is "HD"
            obj.PositionIndex = find(strcmp({obj.MatLabFileName.userdata.locations.catheters.name}, "HD"));
            %finds tensor of "HD" electrode locations
            obj.PosVarPathName = obj.MatLabFileName.userdata.locations.catheters(obj.PositionIndex).egms;
            %finds row index of unipolar raw data where the electrode name is "HD"
            obj.SignalIndex = find(strcmp({obj.MatLabFileName.userdata.epcath_uni_raw.catheters.name}, "HD"));
            %finds 2D array of "HD" electrode signals
            obj.SigVarPathName = obj.MatLabFileName.userdata.epcath_uni_raw.catheters(obj.SignalIndex).egms; %calls unipolar raw data
            %defining position and signal sampling time intervals
            obj.PositionSamplingInterval = PositionSamplingInterval;
            obj.SignalSamplingInterval = SignalSamplingInterval;
            %calculating arrays
            [obj.XCoordArray, obj.YCoordArray, obj.ZCoordArray, obj.PosTimeArray, obj.SignalArray, obj.SignalTimeArray] = readDataMatLab(obj);
        end
        
        %read var
        function [xArr, yArr, zArr, posTimeArr, sigArr, sigTimeArr] = readDataMatLab(obj)
            %signal table of each electrode: column = electrode number
            sigArr = obj.SigVarPathName;
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
    end
end