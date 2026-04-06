clc; clear;

%% ===== SPECS =====
obsInfo = rlNumericSpec([3 1], 'Name','observations'); % [e, d, e_prev]

actInfo = rlNumericSpec([2 1], ...
    'LowerLimit',[-1;-1], ...
    'UpperLimit',[1;1], ...
    'Name','actions');

%% ===== ENV =====
mdl = "ProblemStatewithRL";
agentBlk = "ProblemStatewithRL/RL Agent";

env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

%% ===== ACTOR (3→64→64→2) =====
actorNet = [
    featureInputLayer(3, Name="state")
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    tanhLayer
];

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

%% ===== CRITIC (state+action) =====
% State path
statePath = [
    featureInputLayer(3, Name="state")
    fullyConnectedLayer(64, Name="s_fc1")
    reluLayer(Name="s_relu1")
];

% Action path
actionPath = [
    featureInputLayer(2, Name="action")
    fullyConnectedLayer(64, Name="a_fc1")
];

% Common path
commonPath = [
    additionLayer(2, Name="add")
    reluLayer(Name="c_relu")
    fullyConnectedLayer(1, Name="q_out")
];

criticLG = layerGraph();
criticLG = addLayers(criticLG, statePath);
criticLG = addLayers(criticLG, actionPath);
criticLG = addLayers(criticLG, commonPath);

criticLG = connectLayers(criticLG, "s_relu1", "add/in1");
criticLG = connectLayers(criticLG, "a_fc1", "add/in2");

critic = rlQValueFunction(criticLG, obsInfo, actInfo, ...
    ObservationInputNames="state", ...
    ActionInputNames="action");

%% ===== TD3 OPTIONS (STABLE) =====
agentOpts = rlTD3AgentOptions(...
    SampleTime=0.1, ...
    DiscountFactor=0.99, ...
    MiniBatchSize=128, ...
    ExperienceBufferLength=1e6);

% Target policy smoothing (important for TD3)
agentOpts.TargetPolicySmoothModel.StandardDeviation = 0.2;
agentOpts.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOpts.TargetPolicySmoothModel.UpperLimit = 0.5;

% Exploration noise
agentOpts.ExplorationModel.StandardDeviation = 0.1;

agent = rlTD3Agent(actor, critic, agentOpts);

%% ===== TRAIN OPTIONS =====
trainOpts = rlTrainingOptions(...
    MaxEpisodes=400, ...
    MaxStepsPerEpisode=200, ...
    ScoreAveragingWindowLength=20, ...
    StopTrainingCriteria="AverageReward", ...
    StopTrainingValue=-20, ...
    Verbose=true, ...
    Plots="training-progress");

%% ===== TRAIN =====
trainingStats = train(agent, env, trainOpts);

%% ===== SAVE =====
agentObj = agent;
save('trainedAgent2.mat','agentObj','env');

disp("✅ TD3 agent trained and saved.");