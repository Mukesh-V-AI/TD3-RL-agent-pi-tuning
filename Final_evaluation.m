clc; close all;

%% ===== SETPOINT =====
setpoint = 900;

%% ===== LOAD AGENT (if needed) =====
% Not required for sim(), but safe to keep
load('trainedAgent.mat');

%% ================= BASELINE MODEL =================
simOut_base = sim("ProblemState");

% Extract signal safely
data_base = simOut_base.logsout.getElement('Actual Response');

y_base = data_base.Values.Data;
t_base = data_base.Values.Time;

[OS_b, US_b, Ts_b, Err_b] = get_metrics(y_base, t_base, setpoint);

%% ================= RL MODEL =================
simOut_rl = sim("ProblemStatewithRL");

% Extract signal safely
data_rl = simOut_rl.logsout.getElement('Actual Response');

y_rl = data_rl.Values.Data;
t_rl = data_rl.Values.Time;

[OS_r, US_r, Ts_r, Err_r] = get_metrics(y_rl, t_rl, setpoint);

%% ================= PRINT RESULTS =================
fprintf("\n===== PERFORMANCE COMPARISON =====\n");
fprintf("Baseline → OS=%.2f | Ts=%.2f | Err=%.2f\n", OS_b, Ts_b, Err_b);
fprintf("RL       → OS=%.2f | Ts=%.2f | Err=%.2f\n", OS_r, Ts_r, Err_r);

%% ================= GRAPH 1: RESPONSE =================
figure
plot(t_base, y_base, '--r','LineWidth',2)
hold on
plot(t_rl, y_rl, 'b','LineWidth',2)
yline(setpoint,'--k','Setpoint')

grid on
xlabel('Time (sec)')
ylabel('Output')
title('Baseline vs RL Controller')
legend('Baseline PI','RL PI','Setpoint')

%% ================= GRAPH 2: DEVIATION =================
dev = (y_rl - setpoint)/setpoint * 100;

figure
plot(t_rl, dev,'LineWidth',2)
hold on
yline(15,'--r'); 
yline(-15,'--r');
yline(1,'--g'); 
yline(-1,'--g');

grid on
xlabel('Time (sec)')
ylabel('Deviation (%)')
title('Performance Evaluation (RL Controller)')
legend('Deviation','Overshoot limit','Undershoot limit','Settling band')

%% ================= GRAPH 3: BAR COMPARISON =================
figure

subplot(1,2,1)
bar([OS_b OS_r])
set(gca,'XTickLabel',{'Baseline','RL'})
ylabel('Overshoot (%)')
title('Overshoot Comparison')

subplot(1,2,2)
bar([Ts_b Ts_r])
set(gca,'XTickLabel',{'Baseline','RL'})
ylabel('Time (sec)')
title('Settling Time Comparison')

%% ================= METRIC FUNCTION =================
function [OS, US, Ts, Err] = get_metrics(y, t, setpoint)

% Overshoot
OS = (max(y) - setpoint)/setpoint * 100;

% Undershoot
US = (setpoint - min(y))/setpoint * 100;

% Steady-state error
Err = abs(y(end) - setpoint);

% Settling time (1% band)
threshold = 0.01 * setpoint;
idx = find(abs(y - setpoint) > threshold);

if isempty(idx)
    Ts = 0;
else
    Ts = t(max(idx));
end

end