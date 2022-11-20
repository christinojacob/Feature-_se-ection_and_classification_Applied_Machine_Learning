clear all
close all
clc;
fprintf('\nFEATURE SELECTION TOOLBOX v 6.2 2018 - For Matlab \n');
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath('./Datasets');
addpath(genpath('./lib/drtoolbox'));

% Select a feature selection method from the list
listFS = {'fsasl','dgufs','ufsol'};

[ methodID ] = readInput( listFS );
selection_method = listFS{methodID}; % Selected



M=readtable('D13.csv');
X=table2array(M(:,1:end-1));
Y=table2array(M(:,end));


% Randomly partitions observations into a training set and a test
% set using stratified holdout
P = cvpartition(Y,'Holdout',0.20);

X_train = double( X(P.training,:) );
Y_train = (double( Y(P.training) ));

X_test = double( X(P.test,:) );
Y_test = (double( Y(P.test)));


numF = size(X_train,2);


% feature Selection on training data
switch lower(selection_method)
   
    case 'fsasl'
        options.lambda1 = 1;
        options.LassoType = 'SLEP';
        options.SLEPrFlag = 1;
        options.SLEPreg = 0.01;
        options.LARSk = 5;
        options.LARSratio = 2;
        nClass=2;
        [W, S, A, objHistory] = FSASL(X_train', nClass, options);
        [v,ranking]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
        T=[ranking];
        csvwrite('D:\AML PROJECT\New\FSLib_v7.0.1_2020_2\Datasets\Rankings\fsasl\D13_R_fsasl.csv',T);

    case 'ufsol'
        para.p0 = 'sample';
        para.p1 = 1e6;
        para.p2 = 1e2;
        nClass = 2;
        [~,~,ranking,~] = UFSwithOL(X_train',nClass,para) ;
        T=[ranking];
        csvwrite('D:\AML PROJECT\New\FSLib_v7.0.1_2020_2\Datasets\Rankings\UFSOL\D13_R_ufsol.csv',T);
        
    case 'dgufs'
        
        S = dist(X_train');
        S = -S./max(max(S)); % it's a similarity
        nClass = 2;
        alpha = 0.5;
        beta = 0.9;
        nSel = 2;
        [Y,L,V,Label] = DGUFS(X_train',nClass,S,alpha,beta,nSel);
        [v,ranking]=sort(Y(:,1)+Y(:,2),'descend');
        T=[ranking];
        csvwrite('D:\AML PROJECT\New\FSLib_v7.0.1_2020_2\Datasets\Rankings\DGUFS\D13_R_dgufs.csv',T);
    otherwise
        disp('Unknown method.')
end

