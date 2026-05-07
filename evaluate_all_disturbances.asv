clc; close all;

%% =========================
% LOAD AGENT
%% =========================
load('trainedAgent_SAC.mat','agent');

agentObj = agent;

setpoint = 900;

%% =========================
% TEST CASES
%% =========================
cases = [5 10 15 20 30 99];

results = [];

figure
hold on

%% =========================
% LOOP
%% =========================
for i = 1:length(cases)

    case_id = cases(i);

    assignin('base','case_id',case_id);

    simOut = sim("ProblemStatewithRL");

    y = simOut.y_rl;
    t = simOut.tout;

    [OS, US, Ts, Err] = get_metrics(y,t,setpoint);

    results = [results;
        case_id OS US Ts Err];

    plot(t,y,'LineWidth',2)

end

%% =========================
% GRAPH
%% =========================
yline(setpoint,'--k','Setpoint')

legend("5%","10%","15%","20%","30%","Large Step")

grid on

xlabel("Time")
ylabel("Output")

title("Final RL Performance")

%% =========================
% TABLE
%% =========================
resultsTable = array2table(results,...
    'VariableNames',{'Case','OS','US','Ts','Error'});

disp(resultsTable)

%% =========================
% BAR GRAPH
%% =========================
figure

subplot(2,2,1)
bar(results(:,2))
title("Overshoot")

subplot(2,2,2)
bar(results(:,3))
title("Undershoot")

subplot(2,2,3)
bar(results(:,4))
title("Settling Time")

subplot(2,2,4)
bar(results(:,5))
title("Error")

%% =========================
% METRICS
%% =========================
function [OS, US, Ts, Err] = get_metrics(y,t,sp)

OS = max(0,(max(y)-sp)/sp*100);

idx = find(y > 0.9*sp,1);

if isempty(idx)
    US = 100;
else
    US = max(0,(sp-min(y(idx:end)))/sp*100);
end

Err = abs(y(end)-sp);

tol = 0.01*sp;

idx2 = find(abs(y-sp) > tol);

if isempty(idx2)
    Ts = 0;
else
    Ts = t(max(idx2));
end

end