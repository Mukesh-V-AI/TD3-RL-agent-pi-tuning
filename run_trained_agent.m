clc; close all;

%% =========================
% LOAD AGENT
%% =========================
load('trainedAgent_SAC.mat','agent');

agentObj = agent;

%% =========================
% SELECT DISTURBANCE
%% =========================
case_id = 20;

assignin('base','case_id',case_id);

%% =========================
% RUN MODEL
%% =========================
simOut = sim("ProblemStatewithRL");

%% =========================
% EXTRACT DATA
%% =========================
y = simOut.y_rl;
t = simOut.tout;

%% =========================
% PLOT
%% =========================
figure

plot(t,y,'LineWidth',2)
hold on

yline(900,'--r','Setpoint')

grid on

xlabel('Time (sec)')
ylabel('Output')

title(['RL Controller Response - Disturbance ',num2str(case_id),'%'])