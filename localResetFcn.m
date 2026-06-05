function in = localResetFcn(in)

%% RANDOM DISTURBANCE
dist = randi([5 99]);

in = setVariable(in,...
    'case_id',dist);

%% RESET MEMORY
assignin('base','kp_prev',90);
assignin('base','ki_prev',38);

end