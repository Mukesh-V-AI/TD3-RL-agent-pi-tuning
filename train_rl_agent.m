% =========================================================
% train_rl_agent.m
% Complete RL training script for PI Gain Correction
% =========================================================
% BEFORE running this:
%   1. Open ProblemStatewithRL.slx and confirm:
%      - RL Agent block is present
%      - Observation input = Mux(error, disturbance)
%      - Reward input = reward subsystem output
%      - Two Saturation blocks [0.5, 2.0] on outputs
%      - Two Product blocks for Kp_fixed x Kp_corr
%   2. Run this script in MATLAB
% =========================================================

clear; clc;
fprintf('==============================================\n');
fprintf('  PI Gain RL Training — DDPG Agent\n');
fprintf('==============================================\n\n');

%% ── STEP 1: Parameters ───────────────────────────────────
modelName  = 'ProblemStatewithRL';   % your .slx file name
Ts         = 0.1;                    % sample time (seconds)
Tf         = 200;                    % episode length (seconds)
Kp_fixed   = 1.0;                    % from PI Controller block
Ki_fixed   = 1.0;                    % from PI Controller block
setpoint   = 900;                    % desired RPM

% Put fixed gains in workspace
assignin('base', 'Kp_fixed', Kp_fixed);
assignin('base', 'Ki_fixed', Ki_fixed);
assignin('base', 'setpoint', setpoint);
assignin('base', 'Ts', Ts);

fprintf('Kp_fixed = %.2f,  Ki_fixed = %.2f\n', Kp_fixed, Ki_fixed);
fprintf('Sample time Ts = %.2f s,  Episode = %.0f s\n\n', Ts, Tf);

%% ── STEP 2: Observation specification ───────────────────
% What the agent sees each timestep:
%   [error, disturbance]  →  2 observations
obsInfo = rlNumericSpec([2 1], ...
    'LowerLimit', [-900; 0], ...
    'UpperLimit', [ 900; 1]);
obsInfo.Name   = 'observations';
obsInfo.Description = 'error and disturbance';
fprintf('Observation space: %dx%d  [error, disturbance]\n', ...
    obsInfo.Dimension(1), obsInfo.Dimension(2));

%% ── STEP 3: Action specification ────────────────────────
% What the agent outputs each timestep:
%   [Kp_corr, Ki_corr]  →  2 continuous actions
% Bounded to [0.5, 2.0] — matches Saturation blocks in Simulink
actInfo = rlNumericSpec([2 1], ...
    'LowerLimit', [0.5; 0.5], ...
    'UpperLimit', [2.0; 2.0]);
actInfo.Name        = 'actions';
actInfo.Description = 'Kp_corr and Ki_corr';
fprintf('Action space: %dx%d  [Kp_corr, Ki_corr]  bounds=[0.5, 2.0]\n\n', ...
    actInfo.Dimension(1), actInfo.Dimension(2));

%% ── STEP 4: Create Simulink RL environment ───────────────
fprintf('Creating Simulink RL environment...\n');

% These are the block paths inside your Simulink model
% Adjust if your block names are different
env = rlSimulinkEnv(modelName, ...
    [modelName '/RL Agent'], ...
    obsInfo, actInfo);

% Reset function — sets a random disturbance profile each episode
env.ResetFcn = @(in) localResetFcn(in, modelName);

% Episode length
env.UseFastRestart = 'on';
fprintf('Environment created: %s\n\n', modelName);

%% ── STEP 5: Build actor network (policy) ────────────────
% Actor maps observations → actions
% Architecture: 2 → 64 → 64 → 2 (tanh output scaled to action bounds)

fprintf('Building actor network...\n');
statePath = [
    featureInputLayer(2, 'Normalization', 'none', 'Name', 'obs')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2,  'Name', 'fc_out')
    tanhLayer('Name', 'tanh_out')
    scalingLayer('Name', 'scale_out', 'Scale', 0.75, 'Bias', 1.25)
    % tanh output is [-1,1], scale to [0.5, 2.0]:
    % output = tanh * 0.75 + 1.25  =>  [-0.75+1.25, 0.75+1.25] = [0.5, 2.0]
];

actorNet = dlnetwork(statePath);
actor    = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
               'ObservationInputNames', 'obs');
fprintf('Actor network: 2 -> 64 -> 64 -> 2\n');

%% ── STEP 6: Build critic network (value function) ───────
% Critic maps (observations, actions) → Q-value (scalar)
% Architecture: (2+2) concatenated → 128 → 64 → 1

fprintf('Building critic network...\n');
obsPath  = featureInputLayer(2, 'Normalization','none','Name','obs_in');
actPath  = featureInputLayer(2, 'Normalization','none','Name','act_in');
catLayer = concatenationLayer(1, 2, 'Name', 'cat');
hidLayer = [
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64,  'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1,   'Name', 'fc_out')
];

criticNet = layerGraph();
criticNet = addLayers(criticNet, obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, catLayer);
criticNet = addLayers(criticNet, hidLayer);
criticNet = connectLayers(criticNet, 'obs_in', 'cat/in1');
criticNet = connectLayers(criticNet, 'act_in', 'cat/in2');
criticNet = dlnetwork(criticNet);

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
             'ObservationInputNames', 'obs_in', ...
             'ActionInputNames',      'act_in');
fprintf('Critic network: (2+2) -> 128 -> 64 -> 1\n\n');

%% ── STEP 7: Create DDPG agent ────────────────────────────
fprintf('Creating DDPG agent...\n');

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',              Ts, ...
    'ActorOptimizerOptions',   rlOptimizerOptions('LearnRate', 1e-3), ...
    'CriticOptimizerOptions',  rlOptimizerOptions('LearnRate', 1e-3), ...
    'ExperienceBufferLength',  1e5, ...
    'MiniBatchSize',           128, ...
    'DiscountFactor',          0.99, ...
    'TargetSmoothFactor',      1e-3, ...
    'NoiseOptions',            rlOrnsteinUhlenbeckActionNoise(...
                                   'StandardDeviation', 0.1));

agentObj = rlDDPGAgent(actor, critic, agentOpts);
fprintf('DDPG agent created.\n\n');

% Put agent in workspace so Simulink RL Agent block finds it
assignin('base', 'agentObj', agentObj);
fprintf('agentObj assigned to workspace.\n\n');

%% ── STEP 8: Training options ─────────────────────────────
fprintf('Setting training options...\n');

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',              500, ...
    'MaxStepsPerEpisode',       floor(Tf/Ts), ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria',     'AverageReward', ...
    'StopTrainingValue',        -50, ...     % stop when avg reward > -50
    'SaveAgentCriteria',        'EpisodeReward', ...
    'SaveAgentValue',           -100, ...
    'SaveAgentDirectory',       'saved_agents', ...
    'Verbose',                  true, ...
    'Plots',                    'training-progress');

fprintf('Max episodes: 500,  Steps per episode: %d\n', floor(Tf/Ts));
fprintf('Training will stop when avg reward > -50\n\n');

%% ── STEP 9: Train ────────────────────────────────────────
fprintf('Starting training...\n');
fprintf('(This will take 30-90 minutes. Watch the Training Progress window.)\n\n');

trainingStats = train(agentObj, env, trainOpts);

fprintf('\nTraining complete!\n');
save('training_stats.mat', 'trainingStats', 'agentObj');
fprintf('Saved: training_stats.mat\n\n');

%% ── STEP 10: Test the trained agent ─────────────────────
fprintf('Running final test simulation...\n');

% Large step disturbance test (from problem statement figure)
t_test = [0;   100;  100.01; 500;  500.01; 1000];
d_test = [0.0; 0.0;  0.85;   0.85; 0.25;   0.25];
disturbance_input = [t_test, d_test];
assignin('base', 'disturbance_input', disturbance_input);

simOut = sim(modelName, 'StopTime', '1000');

% Extract and plot results
try
    actual = simOut.logsout.getElement('Actual Response').Values.Data;
    t_out  = simOut.logsout.getElement('Actual Response').Values.Time;
catch
    actual = simOut.yout{1}.Values.Data;
    t_out  = simOut.yout{1}.Values.Time;
end

error_pct = (actual - setpoint) / setpoint * 100;
OS  = max(error_pct);
US  = abs(min(error_pct));

% Settling time
idx = find(abs(error_pct) < 1.0, 1, 'last');
Ts_settle = t_out(idx) if ~isempty(idx) else inf;

fprintf('\n=== FINAL TEST RESULTS ===\n');
fprintf('  Overshoot  : %.2f%%\n', OS);
fprintf('  Undershoot : %.2f%%\n', US);
fprintf('  Final RPM  : %.2f\n',   actual(end));

% Plots
figure('Position', [50 50 1300 700]);
subplot(3,1,1)
plot(t_out, actual, 'b-', 'LineWidth', 1.5); hold on
yline(setpoint, 'r--', 'Setpoint', 'LineWidth', 1)
yline(setpoint*1.20, 'k:', '+20%'); yline(setpoint*0.80, 'k:', '-20%')
ylabel('RPM'); title('RL Agent — Actual Response vs Setpoint'); grid on

subplot(3,1,2)
plot(t_out, error_pct, 'r-', 'LineWidth', 1.5)
yline(0,'k--'); yline(20,'k:','+20%'); yline(-20,'k:','-20%')
ylabel('Error (%)'); title('Percentage Error'); grid on

subplot(3,1,3)
dist_interp = interp1(t_test, d_test, t_out, 'previous', 'extrap');
plot(t_out, dist_interp, 'g-', 'LineWidth', 2)
ylabel('Disturbance'); xlabel('Time (s)'); title('Disturbance Profile'); grid on

sgtitle('RL-tuned PI Controller — Final Test');
saveas(gcf, 'rl_final_test.png');
fprintf('Plot saved: rl_final_test.png\n');

%% ── Local reset function ────────────────────────────────
function in = localResetFcn(in, modelName)
% Called at start of each episode — randomises disturbance profile
% so agent learns to generalise across all disturbance levels

    step_sizes = [0.05, 0.10, 0.15, 0.20, 0.30];
    chosen     = step_sizes(randi(length(step_sizes)));
    
    t_ep = (0:0.1:200)';
    n_steps = round(1/chosen);
    step_dur = 100 / n_steps;
    d_ep = zeros(size(t_ep));
    
    for k = 1:n_steps
        t0 = (k-1)*step_dur; t1 = k*step_dur;
        d_ep(t_ep>=t0 & t_ep<t1) = (k-1)*chosen;
    end
    for k = 1:n_steps
        t0 = 100+(k-1)*step_dur; t1 = 100+k*step_dur;
        d_ep(t_ep>=t0 & t_ep<t1) = 1-(k-1)*chosen;
    end
    
    dist_signal = [t_ep, d_ep];
    in = setVariable(in, 'disturbance_input', dist_signal);
end
