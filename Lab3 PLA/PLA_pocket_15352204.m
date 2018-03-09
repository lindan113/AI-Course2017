
% ʵ���˿ڴ��㷨��F1��Ϊָ��    
% LimitTimes = 8000;

trainVectors = csvread('train.csv');   %ѵ��������
testVectors = csvread('test.csv');     %���Լ�����
[trainRow,trainColumn] = size(trainVectors);  % ��ȡ [�������������������ĳ���+1]
[testRow,testColumn] = size(testVectors);    % ��ȡ [�������������������ĳ���]
LimitTimes = 8000;
w = ones(1,trainColumn);   % ��ʼ��Ȩ������w 
w_best = w;                % ȫ������w
w_next = w ;
max_F1 = 0;
Iteration = 0;              %��������
while true                  %һֱѭ����������������
    Iteration = Iteration + 1; 
    Continue = false;       %�Ƿ���Ҫ��Ҫ�޸�w
    w = w_next ;
    TP = 0;  % Truely postive   ��ȷ��Ԥ��Ϊ+1
    TN = 0;  % Truely negative  ��ȷ��Ԥ��Ϊ-1
    FP = 0;  % Falsely postive   �����Ԥ��Ϊ+1������Ϊ-1
    FN = 0;  % Falsely negative  �����Ԥ��Ϊ+1������Ϊ+1
    Error = 0;
    for i = 1 : trainRow    %��֤ÿһ�������Ƿ�Ԥ����ȷ
        x = [ 1, trainVectors(i,1: trainColumn - 1)]; 
        % x �������,ǰ�油1 ������ȥ����ȷ�� 
        predict = sign( dot(w, x )) ;
        ground_truth = trainVectors(i,trainColumn) ;
        if predict ~= ground_truth
            % Ԥ�����ȷ��
            w_next = w + trainVectors(i,trainColumn) .* x ;
            Continue = true;  %Ԥ�������Ҫ�޸�
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
    % ֹͣ����������ȫ��Ԥ����ȷ or ������������
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
    x = [ 1, testVectors(i,1: testColumn )];   %����-1
    predict = sign( dot(w_best, x )) ;
    testP(i) = predict ;  % ��֤��Ԥ����
end
