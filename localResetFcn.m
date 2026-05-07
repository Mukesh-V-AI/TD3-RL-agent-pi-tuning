function in = localResetFcn(in)

%% =========================================================
% RANDOM DISTURBANCE RESET FUNCTION
%% =========================================================

distCases = [5 10 15 20 30 99];

idx = randi(length(distCases));

case_id = distCases(idx);

in = setVariable(in, ...
    'case_id', ...
    case_id);

fprintf('\n------------------------------------\n');
fprintf('NEW EPISODE DISTURBANCE = %d\n', case_id);
fprintf('------------------------------------\n');

end