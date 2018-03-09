% PLA ԭʼ(���ǲ����ͷ��ʼ��)
% LimitTimes = 10000;
trainVectors = csvread('train.csv');   %ѵ��������
valVectors = csvread('val.csv');       %��֤������
TP = 0;  % Truely postive   ��ȷ��Ԥ��Ϊ+1
TN = 0;  % Truely negative  ��ȷ��Ԥ��Ϊ-1
FP = 0;  % Falsely postive   �����Ԥ��Ϊ+1������Ϊ-1
FN = 0;  % Falsely negative  �����Ԥ��Ϊ+1������Ϊ+1

[trainRow,trainColumn] = size(trainVectors);  % ��ȡ [�������������������ĳ���+1]
[valRow,valColumn] = size(valVectors);        % ��ȡ [�������������������ĳ���+1]
w = ones(1,trainColumn);   % ��ʼ��Ȩ������w 
valP = zeros(1,valRow);     % P ��֤��Ԥ�������� 
LimitTimes = 10000;
Iteration = 0;              %��������
while true                  %һֱѭ����������������
    Iteration = Iteration + 1; 
    Continue = false;       %�Ƿ���Ҫ��Ҫ�޸�w
    for i = 1 : trainRow    %��֤ÿһ�������Ƿ�Ԥ����ȷ
        x = [ 1, trainVectors(i,1: trainColumn - 1)]; 
        % x �������,ǰ�油1 ������ȥ����ȷ�� 
        if sign( dot(w, x )) ~= trainVectors(i,trainColumn)
            % Ԥ�����ȷ��
            w = w + trainVectors(i,trainColumn) .* x ;
            Continue = true;  %Ԥ�������Ҫ�޸�
        end
    end
    if ~Continue || Iteration >= LimitTimes   % 3 * trainRow
    % ֹͣ����������ȫ��Ԥ����ȷ or ��������������2*trainRow��
        break;
    end   
end

Error = 0;
for i = 1 : valRow
    x = [ 1, valVectors(i,1: valColumn - 1)];   
    predict = sign( dot(w, x )) ;
    valP(i) = predict ;  % ��֤��Ԥ����
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
