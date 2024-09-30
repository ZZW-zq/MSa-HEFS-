clear all
clc
% load ORL  
% load Yale    
% load Isolet   
% load BASEHOCK  
% load CLL-SUB-111   
% load USPS   

% load MFD  
% X = C(:,1:size(C,2)-1);
% Y = C(:,size(C,2));

% load SRBCT    
% X = Data(:,2:size(Data,2));
% Y = Data(:,1);
 
% load URBAN1     
% X = URBAN1(:,2:size(URBAN1,2));
% Y = URBAN1(:,1);
 
% load GFE      
% X = data(:,2:size(data,2));
% Y = data(:,1);

% M = csvread('clean1.csv');   
% X = M(:,1:size(M,2)-1);    
% Y = M(:,size(M,2));

M = csvread('musk1.csv');       %The above are 12 datasets used in the experiment
X = M(:,2:size(M,2));
Y = M(:,1);



t0 = clock;
[result, DB] = SAEFS(X, Y, 100);    %results
sum(result(size(result,1),1:size(result,2)-2))  %the number of features
tt1 = etime(clock,t0)  %running time


