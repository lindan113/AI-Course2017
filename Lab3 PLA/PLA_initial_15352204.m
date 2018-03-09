% PLA 原始(但是不会从头开始的)
% LimitTimes = 10000;
trainVectors = csvread('train.csv');   %训练集矩阵
valVectors = csvread('val.csv');       %验证集矩阵
TP = 0;  % Truely postive   正确地预测为+1
TN = 0;  % Truely negative  正确地预测为-1
FP = 0;  % Falsely postive   错误地预测为+1，本来为-1
FN = 0;  % Falsely negative  错误地预测为+1，本来为+1

[trainRow,trainColumn] = size(trainVectors);  % 获取 [样本个数，特征向量的长度+1]
[valRow,valColumn] = size(valVectors);        % 获取 [样本个数，特征向量的长度+1]
w = ones(1,trainColumn);   % 初始化权重向量w 
valP = zeros(1,valRow);     % P 验证集预测结果向量 
LimitTimes = 10000;
Iteration = 0;              %迭代次数
while true                  %一直循环到满足收敛条件
    Iteration = Iteration + 1; 
    Continue = false;       %是否需要需要修改w
    for i = 1 : trainRow    %验证每一个样本是否预测正确
        x = [ 1, trainVectors(i,1: trainColumn - 1)]; 
        % x 增广矩阵,前面补1 ，后面去掉正确答案 
        if sign( dot(w, x )) ~= trainVectors(i,trainColumn)
            % 预测≠正确答案
            w = w + trainVectors(i,trainColumn) .* x ;
            Continue = true;  %预测错误，需要修改
        end
    end
    if ~Continue || Iteration >= LimitTimes   % 3 * trainRow
    % 停止迭代条件：全部预测正确 or 超出迭代次数（2*trainRow）
        break;
    end   
end

Error = 0;
for i = 1 : valRow
    x = [ 1, valVectors(i,1: valColumn - 1)];   
    predict = sign( dot(w, x )) ;
    valP(i) = predict ;  % 验证集预测结果
    ground_truth = valVectors(i,valColumn) ;
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
AccuracyRate = (TP+TN) / valRow
Recall = TP / (TP+FN)
Precision = TP / (TP+FP)
F1 = 2*Precision*Recall / (Precision+Recall)
