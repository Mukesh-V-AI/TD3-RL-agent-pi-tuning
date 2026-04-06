% =========================================================
% add_reward_block.m
% Adds reward subsystem to ProblemStatewithRL.slx
% Run this ONCE before training
% =========================================================

modelName = 'ProblemStatewithRL';
load_system(modelName);

fprintf('Adding reward computation to %s...\n', modelName);

%% ── Add reward subsystem ─────────────────────────────────
% The reward is: -abs(error_pct) - penalty for large overshoot
% This is implemented as a MATLAB Function block

% Add a MATLAB Function block for reward
rewardBlockPath = [modelName '/Reward'];

try
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
              rewardBlockPath, ...
              'Position', [400, -120, 500, -80]);
    
    % Set the MATLAB function code
    rt = sfroot();
    m  = rt.find('-isa','Simulink.BlockDiagram','Name', modelName);
    fc = m.find('-isa','Stateflow.EMChart','Path', rewardBlockPath);
    
    fc.Script = sprintf([...
        'function reward = compute_reward(error, setpoint)\n'...
        '    error_pct = abs(error / setpoint) * 100;\n'...
        '    reward = -error_pct;\n'...
        '    if error_pct > 20\n'...
        '        reward = reward - 50;\n'...
        '    end\n'...
        '    if error_pct > 15\n'...
        '        reward = reward - 20;\n'...
        '    end\n'...
        '    if error_pct < 1.0\n'...
        '        reward = reward + 10;\n'...
        '    end\n'...
        'end\n']);
    
    fprintf('Reward block added at: %s\n', rewardBlockPath);
catch ME
    fprintf('Could not auto-add block: %s\n', ME.message);
    fprintf('Add manually — see instructions below.\n');
end

save_system(modelName);
fprintf('\nModel saved.\n');

%% ── Manual wiring instructions ───────────────────────────
fprintf('\n');
fprintf('============================================\n');
fprintf('MANUAL WIRING CHECKLIST\n');
fprintf('Open ProblemStatewithRL.slx and verify:\n');
fprintf('============================================\n');
fprintf('\n');
fprintf('[1] OBSERVATION INPUT to RL Agent block:\n');
fprintf('    - Add a Mux block (2 inputs)\n');
fprintf('    - Connect: error signal (Sum output) --> Mux input 1\n');
fprintf('    - Connect: disturbance signal         --> Mux input 2\n');
fprintf('    - Connect: Mux output --> RL Agent observation port\n');
fprintf('\n');
fprintf('[2] REWARD INPUT to RL Agent block:\n');
fprintf('    - Add a MATLAB Function block or Fcn block\n');
fprintf('    - Formula: reward = -abs(u(1)/900)*100 - 50*(abs(u(1)/900)>0.20)\n');
fprintf('    - Input u(1) = error signal (from Sum block)\n');
fprintf('    - Connect output --> RL Agent reward port (bottom port)\n');
fprintf('\n');
fprintf('[3] ACTION OUTPUTS from RL Agent:\n');
fprintf('    RL Agent output 1 --> Saturation [0.5,2.0] --> Product (x Kp_fixed)\n');
fprintf('    RL Agent output 2 --> Saturation [0.5,2.0] --> Product (x Ki_fixed)\n');
fprintf('    Product 1 output --> PI Controller P input\n');
fprintf('    Product 2 output --> PI Controller I input\n');
fprintf('\n');
fprintf('[4] RL AGENT BLOCK SETTINGS (double-click it):\n');
fprintf('    Agent: agentObj\n');
fprintf('    Sample time: 0.1\n');
fprintf('    Observation size: [2 1]\n');
fprintf('    Action size: [2 1]\n');
fprintf('\n');
fprintf('[5] SIMULATION SETTINGS:\n');
fprintf('    Solver: ode45 or ode3 (fixed-step for RL)\n');
fprintf('    Fixed-step size: 0.1 (same as Ts)\n');
fprintf('    Stop time: 200 (set by training, will be overridden)\n');
fprintf('\n');
fprintf('Once wiring is confirmed, run: train_rl_agent.m\n');
