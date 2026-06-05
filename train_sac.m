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
curriculumEpisodes = [100 80 80 60 60 50];

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
    LowerLimit=[-2;-2], ...
    UpperLimit=[2;2]);

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
Normalization="zscore",...
Name="state")

fullyConnectedLayer(256,...
Name="actorFC1")

layerNormalizationLayer(...
Name="actorLN1")

leakyReluLayer(0.01,...
Name="actorLRelu1")

fullyConnectedLayer(256,...
Name="actorFC2")

layerNormalizationLayer(...
Name="actorLN2")

leakyReluLayer(0.01,...
Name="actorLRelu2")

fullyConnectedLayer(128,...
Name="actorFC3")

leakyReluLayer(0.01,...
Name="actorLRelu3")
];

%% =====================================================
% MEAN PATH
%% =====================================================
meanPath = [

fullyConnectedLayer(64,...
Name="meanFC")

leakyReluLayer(0.01,...
Name="meanRelu")

fullyConnectedLayer(2,...
Name="mean")
];

%% =====================================================
% STD PATH
%% =====================================================
stdPath = [

fullyConnectedLayer(64,...
Name="stdFC")

leakyReluLayer(0.01,...
Name="stdRelu")

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
"actorLRelu3","meanFC");

actorNet = connectLayers(actorNet,...
"actorLRelu3","stdFC");
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
Normalization="zscore",...
Name="state")

fullyConnectedLayer(256,...
Name="stateFC1")

layerNormalizationLayer(...
Name="stateLN1")

leakyReluLayer(0.01,...
Name="stateRelu1")

fullyConnectedLayer(256,...
Name="stateFC2")

layerNormalizationLayer(...
Name="stateLN2")

leakyReluLayer(0.01,...
Name="stateRelu2")

fullyConnectedLayer(128,...
Name="stateFC3")

leakyReluLayer(0.01,...
Name="stateRelu3")
];

%% =====================================================
% ACTION PATH
%% =====================================================
actionPathC = [

featureInputLayer(2,...
Normalization="none",...
Name="action")

fullyConnectedLayer(128,...
Name="actionFC1")

leakyReluLayer(0.01,...
Name="actionRelu1")
];

%% =====================================================
% COMMON PATH
%% =====================================================
commonPath = [

concatenationLayer(1,2,...
Name="concat")

fullyConnectedLayer(256,...
Name="commonFC1")

leakyReluLayer(0.01,...
Name="commonRelu1")

fullyConnectedLayer(128,...
Name="commonFC2")

leakyReluLayer(0.01,...
Name="commonRelu2")

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
"stateRelu3","concat/in1");

criticNet = connectLayers(criticNet,...
"actionRelu1","concat/in2");

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

agentOpts.TargetSmoothFactor = 5e-3;

agentOpts.ExperienceBufferLength = 2e5;

agentOpts.MiniBatchSize = 128;

agentOpts.NumWarmStartSteps = 3000;

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

    assignin('base','case_id',case_id);        %% change this line -- PROBLEM IS HERE

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
