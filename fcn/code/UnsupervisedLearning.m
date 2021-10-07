%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                   %%%%
%%%%                       Unsupervised learning                       %%%%
%%%%                                                                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Result=UnsupervisedLearning(OutputDataFile, ShapeDataFile, StressDataFile,IdxList_train, IdxList_test,SV_Shape,nNod)
%% Initialization
Result=0;

%% Load Data Files
load(ShapeDataFile)
load(StressDataFile)

%% Shape encoding
%ShapeData_train=ShapeData(:,IdxList_train(randperm(length(IdxList_train)))); %% Training shape data
ShapeData_train=ShapeData(:,IdxList_train); %% Training shape data
ShapeData_test=ShapeData(:,IdxList_test);   %% Testing shape data

MeanShape=mean(ShapeData_train,2);  %% Mean shape
X=ShapeData_train-MeanShape; %% Substract Mean Shape

X=X/sqrt(length(IdxList_train)); %% Don't understand why?

[U, S, V]=svd(X); %%% Singular value decomposition
Lambda=diag(S); %% Singular Values

V123=sum(Lambda(1:SV_Shape).^2)/sum(Lambda.^2); %% Info retained 

PC_count=SV_Shape;
PC=U(:,1:PC_count);
Proj=zeros(nNod*3,PC_count);

for k=1:PC_count
    Proj(:,k)=U(:,k)/Lambda(k);
end

ShapeCode_train=zeros(PC_count,length(IdxList_train));
for k=1:length(IdxList_train)
    temp=ShapeData_train(:,k)-MeanShape;
    c=zeros(1,PC_count);
    for n=1:PC_count
        c(n)=sum(PC(:,n).*temp(:))/Lambda(n);
    end
    ShapeCode_train(:,k)=c;
end

ShapeCode_test=zeros(PC_count,length(IdxList_test));
for k=1:length(IdxList_test)
    temp=ShapeData_test(:,k)-MeanShape;
    c=zeros(1,PC_count);
    for n=1:PC_count
        c(n)=sum(PC(:,n).*temp(:))/Lambda(n);
    end
    ShapeCode_test(:,k)=c;
end

%% 

StressData_train=StressData(:,IdxList_train);
StressData_test=StressData(:,IdxList_test);

%% Save
save(OutputDataFile, 'MeanShape','Proj','ShapeCode_train', 'ShapeCode_test', 'StressData_train','StressData_test', ...
    'OutputDataFile', 'ShapeDataFile', 'StressDataFile',...
    'IdxList_train', 'IdxList_test','V123');
Result=1;



 