function [ OA,AA,kappa,CA ] = SVM( img,train_samples,train_labels,test_SL,test_labels)

[no_lines, no_rows, no_bands] = size(img);
img = reshape(img, no_lines * no_rows, no_bands);
[train_samples,M,m] = scale_func(train_samples);
[img ] = scale_func(img,M,m);
train_labels = train_labels';
test_labels = test_labels';
%%%% Select the paramter for SVM with five-fold cross validation
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);

%%%% Training using a Gaussian RBF kernel
%%% give the parameters of the SVM (Thanks Pedram for providing the
%%% parameters of the SVM)
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 


%%% Train the SVM
model=svmtrain(train_labels,train_samples,parameter);

%%%% SVM Classification
SVMresult = svmpredict(ones(no_lines*no_rows,1),img,model); 

%%%% Evaluation the performance of the SVM
GroudTest = double(test_labels(:,1));
SVMResultTest = SVMresult(test_SL(1,:),:);
[OA,AA,kappa,CA]=confusion(GroudTest,SVMResultTest);


end

