
% 实现了口袋算法，F1作为指标    
% LimitTimes = 8000;

trainVectors = csvread('train.csv');   %训练集矩阵
testVectors = csvread('test.csv');     %测试集矩阵
[trainRow,trainColumn] = size(trainVectors);  % 获取 [样本个数，特征向量的长度+1]
[testRow,testColumn] = size(testVectors);    % 获取 [样本个数，特征向量的长度]
LimitTimes = 8000;
w = ones(1,trainColumn);   % 初始化权重向量w 
w_best = w;                % 全局最优w
w_next = w ;
max_F1 = 0;
Iteration = 0;              %迭代次数
while true                  %一直循环到满足收敛条件
    Iteration = Iteration + 1; 
    Continue = false;       %是否需要需要修改w
    w = w_next ;
    TP = 0;  % Truely postive   正确地预测为+1
    TN = 0;  % Truely negative  正确地预测为-1
    FP = 0;  % Falsely postive   错误地预测为+1，本来为-1
    FN = 0;  % Falsely negative  错误地预测为+1，本来为+1
    Error = 0;
    for i = 1 : trainRow    %验证每一个样本是否预测正确
        x = [ 1, trainVectors(i,1: trainColumn - 1)]; 
        % x 增广矩阵,前面补1 ，后面去掉正确答案 
        predict = sign( dot(w, x )) ;
        ground_truth = trainVectors(i,trainColumn) ;
        if predict ~= ground_truth
            % 预测≠正确答案
            w_next = w + trainVectors(i,trainColumn) .* x ;
            Continue = true;  %预测错误，需要修改
            Error = Error + 1 ;
        end
        
        if predict == 1 && ground_truth == 1
            TP = TP + 1;    
        elseif predict == -1 && ground_truth == -1
            TN = TN + 1;  
        elseif predict == 1 && ground_truth == -1
            FP = FP + 1;        
        else
            FN = FN + 1;   
        end      
    end
    if ~Continue || Iteration >= LimitTimes   
    % 停止迭代条件：全部预测正确 or 超出迭代次数
        break;
    end
    Recall = TP / (TP+FN);
    Precision = TP / (TP+FP);
    F1 = 2*Precision*Recall / (Precision+Recall) ;
    if F1 > max_F1
        w_best = w ;
        max_F1 =  F1 ;
    end    
end

testP = zeros(1,testRow); 
for i = 1 : testRow
    x = [ 1, testVectors(i,1: testColumn )];   %不用-1
    predict = sign( dot(w_best, x )) ;
    testP(i) = predict ;  % 验证集预测结果
end
