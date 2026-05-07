clc; clear; close all;

%% ================= PARAMETERS =================
EPISODES = 150;
MAX_STEPS = 400;
BATCH = 64;
BUFFER_SIZE = 30000;

dt = 0.01;
sp = 900;

dist_list = [0.05 0.1 0.15 0.2 0.3];

gamma = 0.99;
tau = 0.005;
lr = 5e-4;

lambda_os = 1;
lambda_ts = 1;

state_dim = 5;
action_dim = 2;
nq = 32;

%% ================= NETWORKS =================
actor = createActor(state_dim, action_dim);
critic = createIQN(state_dim, action_dim, nq);
target_critic = critic;

%% ================= REPLAY BUFFER =================
buffer = initBuffer(BUFFER_SIZE);

%% ================= TRAIN LOOP =================
for ep = 1:EPISODES

    % ----- RESET -----
    y = 0; y_dot = 0;
    int_e = 0; e_prev = 0;
    t = 0;

    dist = dist_list(randi(numel(dist_list)));

    max_y = sp;
    settled = false;
    Ts = NaN;

    total_reward = 0;

    for step = 1:MAX_STEPS

        % -------- STATE --------
        e = sp - y;
        de = e - e_prev;

        s = [e; de; y; sp; int_e];

        % -------- ACTION --------
        a = predict(actor, dlarray(s,"CB"));
        a = extractdata(a);

        kp = 1.75 + 1.25*a(1);
        ki = 1.75 + 1.25*a(2);

        % -------- CONTROL --------
        int_e = int_e + e*dt;
        u = kp*e + ki*int_e;

        % -------- PLANT --------
        y_ddot = u - y_dot - y;
        y_dot = y_dot + y_ddot*dt;
        y = y + y_dot*dt;

        % disturbance
        y = y - dist*sp*sin(0.5*t);
        t = t + dt;

        % -------- METRICS --------
        max_y = max(max_y,y);

        if ~settled && abs(e) < 0.01*sp
            settled = true;
            Ts = t;
        end

        OS = max(0,(max_y - sp)/sp);
        if isnan(Ts), Ts_val = MAX_STEPS*dt; else, Ts_val = Ts; end

        % -------- REWARD --------
        e_n = e/sp; de_n = de/sp;
        r = -30*abs(e_n) -5*abs(de_n) -0.01*abs(u);
        if abs(e_n)<0.01, r=r+30; end

        total_reward = total_reward + r;

        % -------- NEXT STATE --------
        s2 = [e; de; y; sp; int_e];

        % -------- STORE --------
        buffer = store(buffer,s,a,r,s2);

        e_prev = e;

        %% ================= TRAIN =================
        if buffer.count > BATCH

            [S,A,R,S2] = sample(buffer,BATCH);

            % ----- Critic -----
            q_pred = predict(critic, dlarray([S;A],"CB"));

            a2 = predict(actor, dlarray(S2,"CB"));
            q_next = predict(target_critic, dlarray([S2;a2],"CB"));

            target = R + gamma * mean(q_next,1);

            Lq = mean((q_pred - target).^2,'all');

            % ----- PINN -----
            res = y_ddot + y_dot + y - u;
            Lp = mean(res.^2);

            % ----- CVaR (distributional) -----
            q_sorted = sort(q_pred,1);
            Q = mean(q_sorted(1:floor(0.2*nq),:),'all');

            % ----- CSAC -----
            g_os = OS - 0.15;
            g_ts = Ts_val - 3;

            La = -Q + lambda_os*g_os + lambda_ts*g_ts;

            Loss = Lq + 0.01*Lp + La;

            % ----- BACKPROP -----
            grad = dlgradient(Loss, actor.Learnables);
            actor = dlupdate(@(w,g) w - lr*g, actor, grad);

            % ----- LAMBDA UPDATE -----
            lambda_os = max(0, lambda_os + 0.01*g_os);
            lambda_ts = max(0, lambda_ts + 0.01*g_ts);

            % ----- TARGET UPDATE -----
            target_critic = softUpdate(target_critic, critic, tau);

        end

    end

    fprintf("EP %d | OS %.2f%% | TS %.2f | Reward %.2f\n", ...
        ep, OS*100, Ts_val, total_reward);

end

%% ================= FUNCTIONS =================

function net = createActor(s_dim,a_dim)
net = dlnetwork([
    featureInputLayer(s_dim)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(a_dim)
    tanhLayer]);
end

function net = createIQN(s_dim,a_dim,nq)
net = dlnetwork([
    featureInputLayer(s_dim+a_dim)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(nq)]);
end

function buf = initBuffer(size)
buf.S=[]; buf.A=[]; buf.R=[]; buf.S2=[];
buf.max=size; buf.count=0;
end

function buf = store(buf,s,a,r,s2)
if buf.count < buf.max
    buf.count = buf.count + 1;
    idx = buf.count;
else
    idx = randi(buf.max);
end
buf.S(:,idx)=s;
buf.A(:,idx)=a;
buf.R(idx)=r;
buf.S2(:,idx)=s2;
end

function [S,A,R,S2] = sample(buf,batch)
idx = randi(buf.count,[1 batch]);
S = buf.S(:,idx);
A = buf.A(:,idx);
R = buf.R(idx);
S2 = buf.S2(:,idx);
end

function target = softUpdate(target, net, tau)
target.Learnables.Value = ...
    (1-tau)*target.Learnables.Value + tau*net.Learnables.Value;
end