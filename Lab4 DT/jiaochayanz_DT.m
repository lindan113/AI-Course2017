%% ������ ������֤
%% ��������
trainVectors = csvread('dt_train.csv');   %ѵ��������
%trainVectors = csvread('dt_train.csv');   %ѵ��������
disp('��ȡ�������...');
[trainRow,trainColumn] = size(trainVectors);  % ��ȡ [�������������������ĳ���+1]
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
valids = [sample1;sample2;sample3;sample4;sample5]; %����������

u=20; % ��֦�Ĳ���
correctR = zeros(3,20);
% ��һ����ID3;
% �ڶ�����C4.5
% ��������GINI

for mode = 1:3
    for k=1:u
        inc_node = k ;
        correctFive = zeros(5,1) ;
        for i = 1:5      
            data0 = trainVectors( trains( i,:),1:trainColumn-1 );
            labels0 = trainVectors( trains( i,:) ,trainColumn) ;
            tree0 = make_tree( data0, labels0, inc_node, mode) ;
            
            validdata0 = trainVectors( valids( i,:),1:trainColumn-1 ) ;
            [validdataRow, ~] = size(validdata0);
            ground_truth0 = trainVectors( valids( i,:),trainColumn) ;
            test_predicts0 = use_tree_predict(validdata0, 1:validdataRow, tree0, unique(labels0)); 
            correctFive(i) = length( find(ground_truth0 ==test_predicts0))/validdataRow;
        end
        correct_avg = sum( correctFive )/5 ;
        correctR(mode,k) = correct_avg;
    end
end    
