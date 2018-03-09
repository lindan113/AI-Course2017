%% 大数据 交叉验证
%% 导入数据
trainVectors = csvread('LR_train.csv');   %训练集矩阵
disp('读取数据完毕...');
[trainRow,trainColumn] = size(trainVectors);  % 获取 [样本个数，特征向量的长度+1]
trainRow = trainRow -  mod(trainRow,5);
sample1 = 1: 5 : trainRow;
sample2 = 2: 5 : trainRow;
sample3 = 3: 5 : trainRow;
sample4 = 4: 5 : trainRow;
sample5 = 5: 5 : trainRow;
train1= [ sample2 sample3 sample4 sample5 ];
train2= [ sample1 sample3 sample4 sample5 ];
train3= [ sample1 sample2 sample4 sample5 ];
train4= [ sample1 sample2 sample3 sample5 ];
train5= [ sample1 sample2 sample3 sample4 ];
trains = [train1; train2; train3; train4; train5];
valids = [sample1;sample2;sample3;sample4;sample5]; %样本的索引

alpha = 0.0001 ;
limit = 3000 ;
correctR = zeros(6,6);
correct_avg = zeros(1,6);
Malpha = [ 0.1, 0.01, 0.001, 0.0001 , 0.00001,0.000001 ];
for k=1:6
    correctFive = zeros(5,1) ;
    for i=1:5
        %W0 = LR_makeW( trainVectors( trains( i,:),: ) ,limit, alpha);
        W0 = LR_makeW( trainVectors( trains( i,:),: ) ,limit, Malpha(k));
        validdata0 = trainVectors( valids( i,:),1:trainColumn-1 ) ;
        [validdataRow, ~] = size(validdata0);
        ground_truth0 = trainVectors( valids( i,:),trainColumn) ;
        test_predicts0 = LR_predict(validdata0, W0); 
        correctFive(i) = length( find(ground_truth0 ==test_predicts0))/validdataRow;
        correctR(k,i) = correctFive(i);
    end
    correct_avg(k) = sum( correctFive )/5 ;
    correctFive(k,6) = sum( correctFive )/5 ;
end


