clc;
close all;

%% =====================================================
% LOAD AGENT
%% =====================================================
load('trainedAgent_SAC.mat','agent');

assignin('base','agent',agent);

%% =====================================================
% DISTURBANCE CASES
%% =====================================================
cases = [5 10 15 20 30 99];

%% =====================================================
% SETPOINT
%% =====================================================
sp = 900;

%% =====================================================
% RESULT STORAGE
%% =====================================================
Results = [];

%% =====================================================
% LOOP THROUGH DISTURBANCES
%% =====================================================
for i = 1:length(cases)

    case_id = cases(i);

    assignin('base','case_id',case_id);

    fprintf('\n====================================\n');
    fprintf('TESTING DISTURBANCE = %d\n',case_id);
    fprintf('====================================\n');

    %% RUN SIMULATION
    simOut = sim('ProblemStatewithRL');

    %% GET RESPONSE
    data = simOut.logsout.getElement('Actual Response');

    y = data.Values.Data;

    t = data.Values.Time;

    %% METRICS
    [OS,US,Ts,Err] = get_metrics(y,t,sp);

    %% STORE
    Results = [Results;
        case_id OS US Ts Err];

    %% PLOT
    figure

    plot(t,y,'LineWidth',2)

    hold on

    yline(sp,'--r','Setpoint')

    grid on

    xlabel('Time (sec)')

    ylabel('Response')

    title(['Disturbance = ' num2str(case_id)])

end

%% =====================================================
% FINAL TABLE
%% =====================================================
ResultsTable = array2table(Results, ...
    'VariableNames', ...
    {'Disturbance',...
     'Overshoot',...
     'Undershoot',...
     'SettlingTime',...
     'SteadyStateError'});

disp(' ')
disp('===== FINAL PERFORMANCE TABLE =====')
disp(ResultsTable)

%% =====================================================
% METRIC FUNCTION
%% =====================================================
function [OS,US,Ts,Err] = get_metrics(y,t,sp)

OS = max(0,(max(y)-sp)/sp*100);

US = max(0,(sp-min(y))/sp*100);

Err = abs(y(end)-sp);

%% 1% settling band
band = 0.01*sp;

idx = find(abs(y-sp) > band);

if isempty(idx)

    Ts = 0;

else

    Ts = t(max(idx));

end

end