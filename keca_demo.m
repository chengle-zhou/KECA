%%Chengle Zhou, et al, Kernel Entropy Component Analysis-Based Robust Hyperspectral Image Supervised Classification
close all; clear all; clc
addpath ('.\common')
%% load original image
load(['.\datasets\KSC.mat']);
load(['.\datasets\KSC_gt2.mat']);

%% size of image
img = KSC;
img2 = KSC;
no_class = 13;
[no_row, no_col, no_bands] = size(img);
img = reshape(img, no_row * no_col, no_bands);
GroundT = GroundT';

%% Select training samples and test samples
indexes = train_random_select(GroundT(2,:)); % based on 24 for each class
train_SL = GroundT(:,indexes);
train_samples = img(train_SL(1,:),:);
train_labels = train_SL(2,:);
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = img(test_SL(1,:),:);
GroudTest = test_SL(2,:);

%% Experiment 1 (original SVM classification result)
[ OA_1,AA_1,kappa_1,CA_1 ] = SVM(img2,train_samples,train_labels,test_SL,GroudTest);

%% Add noisy label
load(['.\datasets\Noise_samples_5.mat']); %Example one
load(['.\datasets\Noise_samples_15.mat']);%Example two
%% Experiment 2 (SVM classification result with noisy labels)
training_label = train_label_noise;
training_data = noise_train_data1(:,3:178);
[OA_2,AA_2,kappa_2,CA_2] = SVM_NoisyLabel(img2,training_label,training_data,test_SL,GroudTest);

%% Experiment 3 (using KECA's SVM classification results)
train_data_index = noise_train_data1(:,2);
train_data_record = [train_label_noise,noise_train_data1(:,1),train_data_index];
% Uncertain sample detection based on nuclear entropy analysis
kernel_type =  'RBF_kernel'; 
kernel_pars = 0.12;
Tv = 0.6;
Ntrain = img(train_data_record(:,3),:);
Nlabel = train_data_record(:,1);
training_data = [];
training_label = [];
for i = 1:max(Nlabel)
    Xtrain =  Ntrain(find(i==Nlabel),:);
    Xtrain_nor = Xtrain./repmat(sqrt(sum(Xtrain.*Xtrain)),[size(Xtrain,1) 1]); % unit norm 2
    omega = kernel_matrix(Xtrain_nor,kernel_type, kernel_pars);
    [eigvec, eigval] = eig(omega);
    % Eigenvalue/vector sorting in descending order
    [D,E] = sort_eigenvalues(eigval,eigvec);
    d = diag(D)';
    [sorted_entropy_index,entropy] = ECA(D,E);
    Sigmoid = 1./(1+exp(-entropy));
    sample_index = find(Sigmoid < Tv);
    training_data = [training_data;Xtrain(sample_index,:)];
    training_label = [training_label;i*ones(length(sample_index),1)];
end
%% KECA classification results
[OA_3,AA_3,kappat_3,CAt_3] = My_SVM_Classifier(img2,training_label,training_data,test_SL,GroudTest);

