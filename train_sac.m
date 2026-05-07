clc;
clear;
close all;

%% =========================================================
% MODEL
%% =========================================================
mdl = "ProblemStatewithRL";

open_system(mdl)

agentBlk = mdl + "/RL Agent";

%% =========================================================
% OBSERVATION SPACE
%% =========================================================
obsInfo = rlNumericSpec([5 1], ...
    LowerLimit=-inf*ones(5,1), ...
    UpperLimit= inf*ones(5,1));

obsInfo.Name = "observations";

%% =========================================================
% ACTION SPACE
%% =========================================================
actInfo = rlNumericSpec([2 1], ...
    LowerLimit=[-1;-1], ...
    UpperLimit=[1;1]);

actInfo.Name = "actions";

%% =========================================================
% ENVIRONMENT
%% =========================================================
env = rlSimulinkEnv( ...
    mdl, ...
    agentBlk, ...
    obsInfo, ...
    actInfo);

%% =========================================================
% RANDOM DISTURBANCE RESET
%% =========================================================
env.ResetFcn = @localResetFcn;

%% =========================================================
% ACTOR NETWORK
%% =========================================================
statePath = [

    featureInputLayer(5,...
        Normalization="none",...
        Name="state")

    fullyConnectedLayer(256,...
        Name="fc1")

    reluLayer(Name="relu1")

    fullyConnectedLayer(256,...
        Name="fc2")

    reluLayer(Name="relu2")
];

%% =========================================================
% MEAN PATH
%% =========================================================
meanPath = [

    fullyConnectedLayer(2,...
        Name="mean")
];

%% =========================================================
% STD PATH
%% =========================================================
stdPath = [

    fullyConnectedLayer(2,...
        Name="std")

    softplusLayer(Name="softplus")
];

%% =========================================================
% ACTOR GRAPH
%% =========================================================
actorLG = layerGraph(statePath);

actorLG = addLayers(actorLG,meanPath);
actorLG = addLayers(actorLG,stdPath);

actorLG = connectLayers(actorLG,...
    "relu2","mean");

actorLG = connectLayers(actorLG,...
    "relu2","std");

%% =========================================================
% ACTOR NETWORK
%% =========================================================
actorNet = dlnetwork(actorLG);

%% =========================================================
% SAC ACTOR
%% =========================================================
actor = rlContinuousGaussianActor( ...
    actorNet,...
    obsInfo,...
    actInfo,...
    ActionMeanOutputNames="mean",...
    ActionStandardDeviationOutputNames="softplus");

%% =========================================================
% CRITIC NETWORK
%% =========================================================
statePathC = [

    featureInputLayer(5,...
        Normalization="none",...
        Name="state")

    fullyConnectedLayer(256,...
        Name="c_fc1")

    reluLayer(Name="c_relu1")

    fullyConnectedLayer(256,...
        Name="c_fc2")

    reluLayer(Name="c_relu2")
];

%% =========================================================
% ACTION PATH
%% =========================================================
actionPathC = [

    featureInputLayer(2,...
        Normalization="none",...
        Name="action")

    fullyConnectedLayer(256,...
        Name="a_fc1")
];

%% =========================================================
% COMMON PATH
%% =========================================================
commonPath = [

    additionLayer(2,...
        Name="add")

    reluLayer(Name="common_relu")

    fullyConnectedLayer(1,...
        Name="QValue")
];

%% =========================================================
% CRITIC GRAPH
%% =========================================================
criticLG = layerGraph();

criticLG = addLayers(criticLG,statePathC);
criticLG = addLayers(criticLG,actionPathC);
criticLG = addLayers(criticLG,commonPath);

criticLG = connectLayers(criticLG,...
    "c_relu2","add/in1");

criticLG = connectLayers(criticLG,...
    "a_fc1","add/in2");

%% =========================================================
% CRITIC 1
%% =========================================================
criticNet1 = dlnetwork(criticLG);

critic1 = rlQValueFunction( ...
    criticNet1,...
    obsInfo,...
    actInfo);

%% =========================================================
% CRITIC 2
%% =========================================================
criticNet2 = dlnetwork(criticLG);

critic2 = rlQValueFunction( ...
    criticNet2,...
    obsInfo,...
    actInfo);

%% =========================================================
% SAC OPTIONS
%% =========================================================
agentOpts = rlSACAgentOptions;

agentOpts.SampleTime = 0.01;

agentOpts.DiscountFactor = 0.995;

agentOpts.ExperienceBufferLength = 1e6;

agentOpts.MiniBatchSize = 256;

agentOpts.TargetSmoothFactor = 1e-3;

agentOpts.NumWarmStartSteps = 15000;

%% =========================================================
% CREATE SAC AGENT
%% =========================================================
agent = rlSACAgent( ...
    actor,...
    [critic1 critic2],...
    agentOpts);

%% =========================================================
% TRAINING OPTIONS
%% =========================================================
trainOpts = rlTrainingOptions( ...
    MaxEpisodes=300,...
    MaxStepsPerEpisode=500,...
    ScoreAveragingWindowLength=20,...
    StopTrainingCriteria="none",...
    Verbose=true,...
    Plots="training-progress");

%% =========================================================
% TRAIN AGENT
%% =========================================================
trainingStats = train( ...
    agent,...
    env,...
    trainOpts);

%% =========================================================
% SAVE AGENT
%% =========================================================
save('trainedAgent_SAC.mat','agent');

disp('FINAL PHYSICS-INFORMED SAC TRAINING COMPLETED');