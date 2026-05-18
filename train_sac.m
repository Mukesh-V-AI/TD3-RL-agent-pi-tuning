clc;
clear;
close all;

%% =====================================================
% HYBRID CURRICULUM SAC TRAINING
% LOW SETTLING TIME + ROBUSTNESS
%% =====================================================

%% =====================================================
% MODEL
%% =====================================================
mdl = 'ProblemStatewithRL';

open_system(mdl)

agentBlk = [mdl '/RL Agent'];

%% =====================================================
% STAGED DISTURBANCES
%% =====================================================
curriculumCases = [5 10 15 20 30 99];

%% =====================================================
% EPISODES
%% =====================================================
curriculumEpisodes = [80 60 60 50 40 30];

%% =====================================================
% OBSERVATION SPACE
% [e de y sp ie]
%% =====================================================
obsInfo = rlNumericSpec([5 1]);

obsInfo.Name = 'observations';

%% =====================================================
% ACTION SPACE
%% =====================================================
actInfo = rlNumericSpec([2 1], ...
    LowerLimit=[-1;-1], ...
    UpperLimit=[1;1]);

actInfo.Name = 'actions';

%% =====================================================
% ENVIRONMENT
%% =====================================================
env = rlSimulinkEnv( ...
    mdl, ...
    agentBlk, ...
    obsInfo, ...
    actInfo);

%% =====================================================
% ACTOR NETWORK
%% =====================================================
statePath = [

    featureInputLayer(5,...
    Normalization="none",...
    Name="state")

    fullyConnectedLayer(256,...
    Name="actorFC1")

    reluLayer(...
    Name="actorRelu1")

    fullyConnectedLayer(256,...
    Name="actorFC2")

    reluLayer(...
    Name="actorRelu2")
];

%% =====================================================
% MEAN PATH
%% =====================================================
meanPath = [

    fullyConnectedLayer(2,...
    Name="mean")
];

%% =====================================================
% STD PATH
%% =====================================================
stdPath = [

    fullyConnectedLayer(2,...
    Name="std")

    softplusLayer(...
    Name="softplus")
];

%% =====================================================
% CREATE ACTOR GRAPH
%% =====================================================
actorNet = layerGraph(statePath);

actorNet = addLayers(actorNet,meanPath);

actorNet = addLayers(actorNet,stdPath);

%% =====================================================
% CONNECT ACTOR
%% =====================================================
actorNet = connectLayers(actorNet,...
    "actorRelu2","mean");

actorNet = connectLayers(actorNet,...
    "actorRelu2","std");

%% =====================================================
% DL NETWORK
%% =====================================================
actorNet = dlnetwork(actorNet);

%% =====================================================
% ACTOR
%% =====================================================
actor = rlContinuousGaussianActor( ...
    actorNet, ...
    obsInfo, ...
    actInfo, ...
    ActionMeanOutputNames="mean", ...
    ActionStandardDeviationOutputNames="softplus");

%% =====================================================
% CRITIC STATE PATH
%% =====================================================
statePathC = [

    featureInputLayer(5,...
    Normalization="none",...
    Name="state")

    fullyConnectedLayer(256,...
    Name="stateFC1")

    reluLayer(...
    Name="stateRelu1")

    fullyConnectedLayer(256,...
    Name="stateFC2")
];

%% =====================================================
% ACTION PATH
%% =====================================================
actionPathC = [

    featureInputLayer(2,...
    Normalization="none",...
    Name="action")

    fullyConnectedLayer(256,...
    Name="actionFC1")
];

%% =====================================================
% COMMON PATH
%% =====================================================
commonPath = [

    additionLayer(2,...
    Name="add")

    reluLayer(...
    Name="commonRelu")

    fullyConnectedLayer(1,...
    Name="QValue")
];

%% =====================================================
% CREATE CRITIC GRAPH
%% =====================================================
criticNet = layerGraph();

criticNet = addLayers(criticNet,statePathC);

criticNet = addLayers(criticNet,actionPathC);

criticNet = addLayers(criticNet,commonPath);

%% =====================================================
% CONNECT CRITIC
%% =====================================================
criticNet = connectLayers(criticNet,...
    "stateFC2","add/in1");

criticNet = connectLayers(criticNet,...
    "actionFC1","add/in2");

%% =====================================================
% CRITIC 1
%% =====================================================
criticNet1 = dlnetwork(criticNet);

critic1 = rlQValueFunction( ...
    criticNet1, ...
    obsInfo, ...
    actInfo);

%% =====================================================
% CRITIC 2
%% =====================================================
criticNet2 = dlnetwork(criticNet);

critic2 = rlQValueFunction( ...
    criticNet2, ...
    obsInfo, ...
    actInfo);

%% =====================================================
% SAC OPTIONS
%% =====================================================
agentOpts = rlSACAgentOptions;

agentOpts.SampleTime = 0.01;

agentOpts.DiscountFactor = 0.995;

agentOpts.TargetSmoothFactor = 1e-3;

agentOpts.ExperienceBufferLength = 1e6;

agentOpts.MiniBatchSize = 256;

agentOpts.NumWarmStartSteps = 5000;

%% =====================================================
% SAC AGENT
%% =====================================================
agent = rlSACAgent( ...
    actor, ...
    [critic1 critic2], ...
    agentOpts);

%% =====================================================
% CURRICULUM TRAINING LOOP
%% =====================================================
for stage = 1:length(curriculumCases)

    %% =================================================
    % DISTURBANCE
    %% =================================================
    case_id = curriculumCases(stage);

    assignin('base','case_id',case_id);

    %% =================================================
    % EPISODES
    %% =================================================
    maxEps = curriculumEpisodes(stage);

    %% =================================================
    % DISPLAY
    %% =================================================
    fprintf('\n====================================\n');

    fprintf('TRAINING STAGE %d\n',stage);

    fprintf('DISTURBANCE = %d\n',case_id);

    fprintf('EPISODES    = %d\n',maxEps);

    fprintf('====================================\n');

    %% =================================================
    % TRAINING OPTIONS
    %% =================================================
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes=maxEps, ...
        MaxStepsPerEpisode=500, ...
        ScoreAveragingWindowLength=20, ...
        StopTrainingCriteria="none", ...
        Verbose=true, ...
        Plots="training-progress");

    %% =================================================
    % TRAIN
    %% =================================================
    trainingStats = train( ...
        agent, ...
        env, ...
        trainOpts);

    %% =================================================
    % SAVE STAGE
    %% =================================================
    save('trainedAgent_SAC.mat','agent');

    stageFile = ...
        ['trained_stage_' num2str(case_id) '.mat'];

    save(stageFile,'agent');

    fprintf('\nSTAGE %d COMPLETED\n',stage);

end

%% =====================================================
% FINAL SAVE
%% =====================================================
save('trainedAgent_FINAL.mat','agent');

fprintf('\n====================================\n');

fprintf('FULL TRAINING COMPLETED\n');

fprintf('====================================\n');

