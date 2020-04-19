function [ OA,AA,kappa,CA ] = SVM_NoisyLabel( img,training_label,training_data,test_SL,GroudTest)

[no_lines, no_rows, no_bands] = size(img);
img = reshape(img, no_lines * no_rows, no_bands);
[training_data,M,m] = scale_func(training_data);
[img] = scale_func(img,M,m);

[Ccv2 Gcv2 cv cv_t] = cross_validation_svm(training_label,training_data);
parameter = sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv2,Gcv2); 
model = svmtrain(training_label,training_data,parameter);
SVMresult = svmpredict(ones(no_lines*no_rows,1),img,model); 
SVMResultTest = SVMresult(test_SL(1,:),:);
[OA,AA,kappa,CA] = confusion(GroudTest,SVMResultTest);
end

