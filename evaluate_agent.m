clc;
clear;
close all;

%% =====================================================
% LOAD TRAINED AGENT
%% =====================================================

load('trainedAgent_FINAL.mat');

assignin('base','agentObj',agent);

%% =====================================================
% MODEL
%% =====================================================

mdl = 'ProblemStatewithRL';

open_system(mdl)

%% =====================================================
% TEST CASES
%% =====================================================

cases = [5 10 15 20 30 99];

%% =====================================================
% REQUIREMENTS FROM PROBLEM STATEMENT
%% =====================================================

OS_limit = [15 20 20 25 25 30];

US_limit = [15 20 20 25 25 30];

Ts_limit = [3 5 5 10 10 15];

SST_limit = [8 8 10 10 15 15];

sp = 900;

%% =====================================================
% STORAGE
%% =====================================================

Results = [];

%% =====================================================
% RESPONSE FIGURE
%% =====================================================

figure;
hold on;
grid on;

%% =====================================================
% MAIN LOOP
%% =====================================================

for i = 1:length(cases)

    disturbance = cases(i);

    fprintf('\n====================================\n');
    fprintf('Testing Disturbance = %d%%\n',disturbance);
    fprintf('====================================\n');

    %% ---------------------------------------------
    % DISTURBANCE
    %% ---------------------------------------------

    assignin('base','case_id',disturbance);

    %% ---------------------------------------------
    % RESET VARIABLES
    %% ---------------------------------------------

    assignin('base','kp_prev',85);

    assignin('base','ki_prev',35);

    clear gainScheduler

    %% ---------------------------------------------
    % SIMULATE
    %% ---------------------------------------------

    simOut = sim(mdl);

    %% ---------------------------------------------
    % RESPONSE
    %% ---------------------------------------------

    sig = simOut.logsout.getElement( ...
        'Actual Response');

    y = sig.Values.Data;

    t = sig.Values.Time;

    %% =================================================
% STEPINFO METRICS
%% =================================================

S = stepinfo( ...
    y, ...
    t, ...
    sp, ...
    'SettlingTimeThreshold',0.01);

Overshoot = S.Overshoot;

Undershoot = S.Undershoot;

RiseTime = S.RiseTime;

SettlingTime = S.SettlingTime;

%% =================================================
% STEADY STATE TIME (0.75%)
%% =================================================

ssTol = 0.0075*sp;

SteadyStateTime = NaN;

for k = 1:length(y)

    if all(abs(y(k:end)-sp) <= ssTol)

        SteadyStateTime = t(k);

        break;

    end

end

if isnan(SteadyStateTime)

    SteadyStateTime = t(end);

end

%% =================================================
% STEADY STATE ERROR
%% =================================================

SSE = abs(y(end)-sp);

    passOS = Overshoot <= OS_limit(i);

passUS = Undershoot <= US_limit(i);

passTs = SettlingTime <= Ts_limit(i);

passSST = SteadyStateTime <= SST_limit(i);

if passOS && passUS && ...
   passTs && passSST

    Status = "PASS";

else

    Status = "FAIL";

end

    %% =================================================
    % STORE RESULTS
    %% =================================================

    Results = [Results;
    disturbance ...
    Overshoot ...
    Undershoot ...
    RiseTime ...
    SettlingTime ...
    SteadyStateTime ...
    SSE];

    %% =================================================
    % STORE STATUS
    %% =================================================

    StatusArray{i} = Status;

    %% =================================================
    % PLOT
    %% =================================================

    plot(t,y,'LineWidth',2);

end

%% =====================================================
% SETPOINT
%% =====================================================

yline(sp,'k--','Setpoint');

yline(1.20*sp,...
    'r--',...
    'OS Limit (20%)');

yline(0.80*sp,...
    'r--',...
    'US Limit (20%)');

yline(1.01*sp,...
    'g--',...
    '+1%');

yline(0.99*sp,...
    'g--',...
    '-1%');

yline(1.0075*sp,...
    'm--',...
    '+0.75%');

yline(0.9925*sp,...
    'm--',...
    '-0.75%');

xlabel('Time (sec)');
ylabel('Response');

title('Adaptive PI Controller Responses');

legend( ...
'5%', ...
'10%', ...
'15%', ...
'20%', ...
'30%', ...
'99%', ...
'Setpoint', ...
'OS Limit', ...
'US Limit', ...
'+1%', ...
'-1%', ...
'+0.75%', ...
'-0.75%');

%% =====================================================
% RESULTS TABLE
%% =====================================================

ResultsTable = table( ...
    Results(:,1), ...
    Results(:,2), ...
    Results(:,3), ...
    Results(:,4), ...
    Results(:,5), ...
    Results(:,6), ...
    Results(:,7), ...
    string(StatusArray)', ...
    'VariableNames', ...
    {'Disturbance',...
     'Overshoot',...
     'Undershoot',...
     'RiseTime',...
     'SettlingTime',...
     'SteadyStateTime',...
     'SSE',...
     'Status'});

%% =====================================================
% DISPLAY
%% =====================================================

fprintf('\n');
fprintf('=============================================\n');
fprintf('        FINAL EVALUATION RESULTS\n');
fprintf('=============================================\n\n');

disp(ResultsTable);
ConstraintTable = table( ...
    cases', ...
    OS_limit', ...
    Results(:,2), ...
    US_limit', ...
    Results(:,3), ...
    Ts_limit', ...
    Results(:,5), ...
    SST_limit', ...
    Results(:,6), ...
    string(StatusArray)', ...
    'VariableNames', ...
    {'Disturbance',...
     'OS_Needed',...
     'OS_Obtained',...
     'US_Needed',...
     'US_Obtained',...
     'Ts_Needed',...
     'Ts_Obtained',...
     'SST_Needed',...
     'SST_Obtained',...
     'Status'});

fprintf('\n');
fprintf('====================================\n');
fprintf('NEEDED vs OBTAINED\n');
fprintf('====================================\n');

disp(ConstraintTable);

%% =====================================================
% OVERSHOOT COMPARISON
%% =====================================================

figure;

bar(cases,Results(:,2));

grid on;

xlabel('Disturbance (%)');
ylabel('Overshoot (%)');

title('Overshoot Comparison');

%% =====================================================
% UNDERSHOOT COMPARISON
%% =====================================================

figure;

bar(cases,Results(:,3));

grid on;

xlabel('Disturbance (%)');
ylabel('Undershoot (%)');

title('Undershoot Comparison');

%% =====================================================
% RISE TIME COMPARISON
%% =====================================================

figure;

bar(cases,Results(:,4));

grid on;

xlabel('Disturbance (%)');
ylabel('Rise Time (sec)');

title('Rise Time Comparison');

%% =====================================================
% SETTLING TIME COMPARISON
%% =====================================================

figure;

bar(cases,Results(:,5));

grid on;

xlabel('Disturbance (%)');
ylabel('Settling Time (sec)');

title('Settling Time Comparison');

%% =====================================================
% STEADY STATE ERROR
%% =====================================================

figure;

bar(cases,Results(:,6));

hold on;

yline(10,...
    'r--',...
    'Requirement');

grid on;

xlabel('Disturbance (%)');

ylabel('Steady State Time (sec)');

title('Steady State Time Comparison');


