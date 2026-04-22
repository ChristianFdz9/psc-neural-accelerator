%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSC Neural Accelerator - Training Pipeline
%
% This script trains and validates a feedforward neural network that
% approximates the optimal switching policy in the PSC framework.
%
% Reference:
%   "Predictive-Switching Control of Stochastic Gene Regulatory Networks:
%    A Contractive PIDE Framework" (2026)
%
% Authors:
%   C. Fernández, M. Pájaro, G. Szederkényi, I. Otero-Muras
%
% -------------------------------------------------------------------------
% USAGE:
%   - Place 'PSC_Master_Dataset.mat' in the working directory.
%   - The dataset must contain:
%         X_raw : [N x d] feature matrix
%         Y_raw : [N x m] binary targets
%   - Run this script to train and export the neural accelerator.
%
% OUTPUT:
%   - Trained model: PSC_Accelerator.mat
%
% NOTES:
%   - Strict hold-out validation (no data leakage)
%   - Z-score normalization computed on training set only
%   - Deterministic execution via fixed RNG seed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ------------------------------------------------------------------------
% 0. INITIALIZATION
% -------------------------------------------------------------------------
clear; close all; clc;
rng(42, 'twister');

fprintf('--- PSC Neural Accelerator Training ---\n');

%% ------------------------------------------------------------------------
% 1. CONFIGURATION
% -------------------------------------------------------------------------
cfg.masterFile = 'PSC_Master_Dataset.mat';
cfg.testRatio  = 0.15;     % Hold-out test set
cfg.kFolds     = 5;        % Cross-validation folds

cfg.arch       = [20 10];  % Network architecture
cfg.trainFcn   = 'trainlm';
cfg.epochs     = 2000;
cfg.maxFail    = 20;

cfg.outputDim  = 3;        % Target dimension

%% ------------------------------------------------------------------------
% 2. DATA LOADING
% -------------------------------------------------------------------------
if ~exist(cfg.masterFile, 'file')
    error('Dataset %s not found. Run data consolidation first.', cfg.masterFile);
end

fprintf('Loading dataset... ');
load(cfg.masterFile);
fprintf('Done.\n');

N_total = size(X_raw, 1);
fprintf('Total samples: %d\n', N_total);

%% ------------------------------------------------------------------------
% 3. HOLD-OUT PARTITION
% -------------------------------------------------------------------------
% Separation performed prior to any transformation (prevents data leakage)

idx_perm  = randperm(N_total);
n_test    = floor(cfg.testRatio * N_total);

idx_test  = idx_perm(1:n_test);
idx_train = idx_perm(n_test+1:end);

X_train_raw = X_raw(idx_train,:);
Y_train     = Y_raw(idx_train,:);

X_test_raw  = X_raw(idx_test,:);
Y_test      = Y_raw(idx_test,:);

%% ------------------------------------------------------------------------
% 4. NORMALIZATION (TRAIN SET ONLY)
% -------------------------------------------------------------------------
[inputs_train_n, ps_in] = mapstd(X_train_raw');
targets_train           = Y_train';

inputs_test_n = mapstd('apply', X_test_raw', ps_in);
targets_test  = Y_test';

%% ------------------------------------------------------------------------
% 5. K-FOLD CROSS-VALIDATION
% -------------------------------------------------------------------------
% Architecture validation over the development set

N_dev    = size(X_train_raw, 1);
idx_rand = randperm(N_dev);
fold_ids = floor(linspace(1, cfg.kFolds + 1, N_dev + 1));

indices = zeros(N_dev, 1);
indices(idx_rand) = fold_ids(1:end-1);

cv_stats = [];

fprintf('\n--- %d-Fold Cross-Validation ---\n', cfg.kFolds);

for k = 1:cfg.kFolds

    val_idx   = (indices == k);
    train_idx = ~val_idx;

    X_cv_train = inputs_train_n(:, train_idx);
    Y_cv_train = targets_train(:, train_idx);
    X_cv_val   = inputs_train_n(:, val_idx);
    Y_cv_val   = targets_train(:, val_idx);

    net_cv = feedforwardnet(cfg.arch, cfg.trainFcn);
    net_cv.layers{1:2}.transferFcn = 'tansig';
    net_cv.layers{3}.transferFcn   = 'satlins';

    net_cv.trainParam.epochs   = cfg.epochs;
    net_cv.trainParam.max_fail = cfg.maxFail;
    net_cv.divideFcn           = 'dividetrain';

    [net_cv, ~] = train(net_cv, X_cv_train, Y_cv_train, 'useParallel', 'no');

    % Projection onto admissible set {0,1}
    Y_val_raw  = net_cv(X_cv_val);
    Y_val_pred = round(max(0, min(1, Y_val_raw)));

    acc = mean(all(Y_val_pred == Y_cv_val, 1));
    cv_stats = [cv_stats; acc];

    fprintf(' Fold %d | Exact Match: %6.2f%%\n', k, acc*100);
end

fprintf('CV Mean: %.2f%% | Std: %.2f\n', ...
    mean(cv_stats)*100, std(cv_stats)*100);

%% ------------------------------------------------------------------------
% 6. FINAL MODEL TRAINING
% -------------------------------------------------------------------------
fprintf('\n--- Training Final Model ---\n');

bestNet = feedforwardnet(cfg.arch, cfg.trainFcn);
bestNet.layers{1:2}.transferFcn = 'tansig';
bestNet.layers{3}.transferFcn   = 'satlins';

bestNet.inputs{1}.processFcns = {};
bestNet.trainParam.epochs     = cfg.epochs;
bestNet.trainParam.max_fail   = cfg.maxFail;

[bestNet, tr] = train(bestNet, inputs_train_n, targets_train);

%% ------------------------------------------------------------------------
% 7. TEST EVALUATION
% -------------------------------------------------------------------------
Y_final_pred_raw = bestNet(inputs_test_n);
Y_final_pred_bin = round(max(0, min(1, Y_final_pred_raw)));

exact_match  = mean(all(Y_final_pred_bin == targets_test, 1)) * 100;
bit_accuracy = mean(Y_final_pred_bin == targets_test, 'all') * 100;

fprintf('\n--- TEST PERFORMANCE ---\n');
fprintf(' Exact Match: %6.2f%%\n', exact_match);
fprintf(' Bit Accuracy: %6.2f%%\n', bit_accuracy);

%% ------------------------------------------------------------------------
% 8. MODEL EXPORT
% -------------------------------------------------------------------------
saveName = 'PSC_Accelerator.mat';

stats.model_type       = 'Neural Accelerator';
stats.training_date    = datetime('now');
stats.architecture     = cfg.arch;
stats.cv_mean_accuracy = mean(cv_stats);
stats.test_exact_match = exact_match;
stats.normalization    = ps_in;

save(saveName, 'bestNet', 'ps_in', 'stats');

fprintf('\nModel saved to: %s\n', saveName);

%% ------------------------------------------------------------------------
% 9. DIAGNOSTICS
% -------------------------------------------------------------------------
figure('Name','Learning Curve','Color','w');
plotperform(tr);
grid on;
title('Training Performance (MSE)');

figure('Name','Confusion Matrix','Color','w');
confusionchart(...
    targets_test(:), ...
    Y_final_pred_bin(:), ...
    'Title','Bit-wise Classification', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');