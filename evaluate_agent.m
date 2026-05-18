clc;
close all;
clear;

%% =====================================================
% LOAD TRAINED AGENT
%% =====================================================
load('trainedAgent_FINAL.mat','agent');

assignin('base','agentObj',agent);

%% =====================================================
% DISTURBANCE CASES
%% =====================================================
cases = [5 10 15 20 30 99];

%% =====================================================
% SETPOINT
%% =====================================================
sp = 900;

%% =====================================================
% STORAGE
%% =====================================================
Results = [];

AllResponses = {};

AllTimes = {};

%% =====================================================
% MAIN LOOP
%% =====================================================
for i = 1:length(cases)

    %% =================================================
    % DISTURBANCE
    %% =================================================
    case_id = cases(i);

    assignin('base','case_id',case_id);

    fprintf('\n====================================\n');

    fprintf('TESTING DISTURBANCE = %d\n',case_id);

    fprintf('====================================\n');

    %% =================================================
    % RUN SIMULATION
    %% =================================================
    simOut = sim('ProblemStatewithRL');

    %% =================================================
    % RESPONSE
    %% =================================================
    data = simOut.logsout.getElement('Actual Response');

    y = data.Values.Data;

    t = data.Values.Time;

    %% =================================================
    % STORE RESPONSES
    %% =================================================
    AllResponses{i} = y;

    AllTimes{i} = t;

    %% =================================================
    % METRICS
    %% =================================================
    [OS,US,Ts,Err] = get_metrics(y,t,sp);

    %% =================================================
    % STORE RESULTS
    %% =================================================
    Results = [Results;
        case_id OS US Ts Err];

    %% =================================================
    % INDIVIDUAL RESPONSE GRAPH
    %% =================================================
    figure

    plot(t,y,'LineWidth',2)

    hold on

    yline(sp,'--r','Setpoint')

    grid on

    xlabel('Time (sec)')

    ylabel('Response')

    title(['RL Response | Disturbance = ' num2str(case_id)])

    legend('Response','Setpoint')

end

%% =====================================================
% RESULTS TABLE
%% =====================================================
ResultsTable = array2table(Results,...
    'VariableNames',...
    {'Disturbance',...
     'Overshoot',...
     'Undershoot',...
     'SettlingTime',...
     'SteadyStateError'});

disp(' ')

disp('===== FINAL PERFORMANCE TABLE =====')

disp(ResultsTable)

%% =====================================================
% ALL RESPONSES IN ONE GRAPH
%% =====================================================
figure

hold on

for i = 1:length(cases)

    plot(AllTimes{i},...
         AllResponses{i},...
         'LineWidth',2)

end

yline(sp,'--k','Setpoint')

grid on

xlabel('Time (sec)')

ylabel('Response')

title('All Disturbance Responses')

legend('5%',...
       '10%',...
       '15%',...
       '20%',...
       '30%',...
       '99%',...
       'Setpoint')

%% =====================================================
% SETTLING TIME GRAPH
%% =====================================================
figure

bar(Results(:,4))

grid on

xticklabels(string(cases))

xlabel('Disturbance (%)')

ylabel('Settling Time (sec)')

title('Settling Time')

%% =====================================================
% OVERSHOOT GRAPH
%% =====================================================
figure

bar(Results(:,2))

grid on

xticklabels(string(cases))

xlabel('Disturbance (%)')

ylabel('Overshoot (%)')

title('Overshoot')

%% =====================================================
% UNDERSHOOT GRAPH
%% =====================================================
figure

bar(Results(:,3))

grid on

xticklabels(string(cases))

xlabel('Disturbance (%)')

ylabel('Undershoot (%)')

title('Undershoot')

%% =====================================================
% STEADY-STATE ERROR GRAPH
%% =====================================================
figure

bar(Results(:,5))

grid on

xticklabels(string(cases))

xlabel('Disturbance (%)')

ylabel('Steady-State Error')

title('Steady-State Error')

%% =====================================================
% METRIC FUNCTION
%% =====================================================
function [OS,US,Ts,Err] = get_metrics(y,t,sp)

%% =====================================================
% OVERSHOOT
%% =====================================================
OS = max(0,(max(y)-sp)/sp*100);

%% =====================================================
% UNDERSHOOT
% Ignore startup transient
%% =====================================================
reachIdx = find(y >= 0.98*sp,1);

if isempty(reachIdx)

    US = 0;

else

    y_after = y(reachIdx:end);

    us_val = min(y_after);

    US = max(0,(sp-us_val)/sp*100);

end

%% =====================================================
% STEADY STATE ERROR
%% =====================================================
Err = abs(y(end)-sp);

%% =====================================================
% SETTLING TIME
%% =====================================================
band = 0.01*sp;

idx = find(abs(y-sp) > band);

if isempty(idx)

    Ts = 0;

else

    Ts = t(max(idx)) ;

end

end