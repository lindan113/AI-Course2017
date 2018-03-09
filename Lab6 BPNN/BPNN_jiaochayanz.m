
trainVectors_raw = csvread('BPNN_train.csv');   %训练集矩阵
[trainRow,trainColumn] = size(trainVectors_raw);  % 获取 [样本个数，特征向量的长度+1]
% trainVectors = trainVectors_raw ;
xMIN = -1 ;
xMAX = 5 ;
[trainVectors,PS] = mapminmax(trainVectors_raw(:,1:trainColumn-1)' ,xMIN,xMAX);
trainVectors = [trainVectors', trainVectors_raw(:,trainColumn) ] ;

H = round(log2(trainRow)) ;
max_iter = 3000;
step = 0.000001;

trainRow = trainRow -  mod(trainRow,5);
sample_num = trainRow/5 ;
randomRow = randperm(trainRow);
sample1 = randomRow(1: sample_num);
sample2 = randomRow(sample_num+1: 2*sample_num);
sample3 = randomRow(2*sample_num+1: 3*sample_num);
sample4 = randomRow(3*sample_num+1: 4*sample_num);
sample5 = randomRow(4*sample_num+1: 5*sample_num);
train1= [ sample2 sample3 sample4 sample5 ];
train2= [ sample1 sample3 sample4 sample5 ];
train3= [ sample1 sample2 sample4 sample5 ];
train4= [ sample1 sample2 sample3 sample5 ];
train5= [ sample1 sample2 sample3 sample4 ];
trains = [train1; train2; train3; train4; train5];
valids = [sample1;sample2;sample3;sample4;sample5]; %样本的索引

X = trainVectors( trains( 1,:), 1:trainColumn-1 );
t = trainVectors( trains( 1,:), trainColumn);

v_truth = trainVectors( valids( 1,:),trainColumn) ;
v_X = trainVectors( valids( 1,:),1:trainColumn-1 ) ;
% output = BPNN_training( X ,t , H, max_iter,step);
%output = BPNN_ReLU( X ,t ,v_X, v_truth, H, max_iter,step);
output = BPNN( X ,t ,v_X, v_truth, H, max_iter,step);
plot(1:max_iter,output.cost,'b-',1:max_iter,output.cost_val,'g-')
legend('training loss','val loss')

% startD = 1 ;
% N = 20 * 24 ;
% endD = startD + N -1  ;
% twenty_X = trainVectors(startD:endD , 1:trainColumn-1 ) ;
% twenty_t = trainVectors(startD:endD , trainColumn ) ;
% twenty_X= [ ones(N,1) , twenty_X ] ;
% twenty_predicts = zeros(N,1);
% for i=1:N
%         ho = sigmoid( twenty_X(i,:) * output.W' ) ;
%         h_t = [1,ho ];
%         twenty_predicts(i) = round(h_t * output.wo');  
%         if twenty_predicts(i) < 0 
%             twenty_predicts(i) = 1 ;
%         end       
% end 
% 
% plot(1:N,twenty_t,'m-',1:N,twenty_predicts,'b-') 
% legend('truth','predict')
