clc;
close all;

%% =====================================================
% LOAD TRAINED AGENT
%% =====================================================
load('trainedAgent_FINAL.mat','agent');

%% =====================================================
% ASSIGN TO SIMULINK
%% =====================================================
assignin('base','agentObj',agent);

%% =====================================================
% TEST DISTURBANCE
%% =====================================================
case_id = 30;

assignin('base','case_id',case_id);

%% =====================================================
% RUN SIMULATION
%% =====================================================
simOut = sim('ProblemStatewithRL');

%% =====================================================
% GET RESPONSE
%% =====================================================
data = simOut.logsout.getElement('Actual Response');

y = data.Values.Data;

t = data.Values.Time;

%% =====================================================
% SETPOINT
%% =====================================================
sp = 900;

%% =====================================================
% PLOT
%% =====================================================
figure

plot(t,y,'LineWidth',2)

hold on

yline(sp,'--r','Setpoint')

grid on

xlabel('Time (sec)')

ylabel('Response')

title(['RL Response | Disturbance = ' num2str(case_id)])

legend('Response','Setpoint')

run('evaluate_agent.m');