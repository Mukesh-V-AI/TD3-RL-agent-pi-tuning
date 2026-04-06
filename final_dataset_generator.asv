% ==========================================================
% FINAL DATASET GENERATION (MAT + SIMULINK COMBINED)
% ==========================================================

clc;
clear;

%% =========================
% STEP 1: LOAD GIVEN DATA
% =========================

data = load('Simulink_Data (1).mat');
ds_mat = data.SimulinkData;

sig_error = ds_mat.get('Error');
sig_dist  = ds_mat.get('Disturbance A');

error_mat = sig_error.Values.Data;
dist_mat  = sig_dist.Values.Data;

% Input from MAT
X_mat = [error_mat dist_mat];

% Generate labels (initial heuristic)
Kp_corr_mat = 1 + 0.1 * abs(error_mat) + 0.05 * dist_mat;
Ki_corr_mat = 1 + 0.05 * abs(error_mat) + 0.02 * dist_mat;

Y_mat = [Kp_corr_mat Ki_corr_mat];

%% =========================
% STEP 2: SIMULATION DATA
% =========================

Kp_range = 0.5:0.5:3;
Ki_range = 0.1:0.2:1.5;

X_sim = [];
Y_sim = [];

for kp = Kp_range
for ki = Ki_range


    assignin('base','Kp_fixed',kp);
    assignin('base','Ki_fixed',ki);
    
    simOut = sim('ProblemState.mdl');
    
    ds = simOut.logsout;
    
    sig_error = ds.get('Error');
    sig_dist  = ds.get('Disturbance A');
    
    error = sig_error.Values.Data;
    dist  = sig_dist.Values.Data;
    
    % Use mean values as state
    X_sim = [X_sim; mean(error), mean(dist)];
    
    % Label = gains used
    Y_sim = [Y_sim; kp, ki];
    
end


end

% Convert to correction factors
Kp_corr_sim = Y_sim(:,1) / mean(Kp_range);
Ki_corr_sim = Y_sim(:,2) / mean(Ki_range);

Y_sim_final = [Kp_corr_sim Ki_corr_sim];

%% =========================
% STEP 3: COMBINE DATA
% =========================

X = [X_mat; X_sim];
Y = [Y_mat; Y_sim_final];

%% =========================
% STEP 4: SAVE DATASET
% =========================

data_final = [X Y];

headers = {'error','disturbance','Kp_corr','Ki_corr'};
T = array2table(data_final, 'VariableNames', headers);

writetable(T, 'Final_dataset.csv');

disp('Final dataset created successfully');
