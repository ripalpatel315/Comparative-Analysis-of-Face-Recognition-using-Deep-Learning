clc;
close all;
clear all;

% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','DigitDataset');
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(400,20);

load('ORL_32x32.mat');


%===========================================
faceW = 32; 
faceH = 32; 
numPerLine = 10; 
ShowLine = 2; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]); 
  	end 
end 

imagesc(Y);colormap(gray);
%===========================================

% for i = 1:20
%     subplot(4,5,i);
%     I1=fea(i,:);
%     I=imresize(I1,[32 32]);
%     imshow(uint8(I));
% end

% labelCount = countEachLabel(gnd);
% img = readimage(imds,1);
 %size();
for i=1:40
    A(i,:)=[i 10];
end
labelCount = array2table(A,'VariableNames',{'Label','Count'});


% 
% numTrainFiles = 750;
% [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

layers = [
    imageInputLayer([32 32 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(40)
    softmaxLayer
    classificationLayer];
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.001, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', 'Plots', 'training-progress');
[r c]=size(fea);

%%%%%%%%%%%%%%%%%%%%%%%
%features
p=1;
fea=fea';
fea1=reshape(fea,[32 32 400]);
for i=0:39
    X_train(:,:,p:p+7)=fea1(:,:,(1+10*i):(8+10*i));
    p=p+8;
end

p=1;
for i=0:39
    X_test(:,:,p:p+1)=fea1(:,:,(9+10*i):(10+10*i));
    p=p+2;
end
%%%%%%%%%%%%%%%%%%%%%%%%
%labels
p=1;
for i=0:39
    T_train(p:p+7)=gnd((1+10*i):(8+10*i));
    p=p+8;
end

p=1;
for i=0:39
    T_test(p:p+1)=gnd((9+10*i):(10+10*i));
    p=p+2;
end

B = categorical(T_train');
X_train=reshape(X_train,[32 32 1 320]);
net = trainNetwork(X_train,B,layers,options);
    
% net = trainNetwork(imdsTrain,layers,options);
X_test=reshape(X_test,[32 32 1 80]);
 YPred = classify(net,X_test);
 
 T = categorical(T_test);
 accuracy = sum(T' == YPred)/numel(T_test);
% YValidation = imdsValidation.Labels;
% 
% accuracy = sum(YPred == YValidation)/numel(YValidation)
