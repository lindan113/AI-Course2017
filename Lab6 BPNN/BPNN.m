
function output = BPNN(X, t, v_X, v_truth,H, max_iter,step)
% X -- 输入训练数据样本特征数据矩阵，N行,即N个样本，n列，即n个特征，
% t -- 输入训练数据的标签，N×1
% v_X -- 输入验证数据样本特征数据矩阵
% v_truth -- 输入验证数据的标签
% H -- 隐藏层的

[N,n] = size(X);
% N -- 训练样本的样本数
% n -- 训练样本的特征个数
v_N = size(v_X, 1) ;
% N -- 验证集样本的样本数
X = [ ones(N,1) , X ] ;
% 增广矩阵

if N~= length(t)
    error('inconsistent sample size');
end

%% 随机初始化 
W = rand(H,n+1);        % W --  输入到隐藏层的权重（包括偏置）
wo = rand(1,H+1);       % wo -- 隐藏层到输出层的权重（包括偏置）  
% W = randn(H,n+1);        % W --  输入到隐藏层的权重（包括偏置）
% wo = randn(1,H+1);       % wo -- 隐藏层到输出层的权重（包括偏置）  
iter = 0;       % 迭代计数
cost_training = zeros(1,max_iter);  % training loss
cost_val = zeros(1,max_iter);       % validation loss
while iter < max_iter       
    delta_W = zeros(H,n+1);     % 初始化Δ_W=0
    delta_wo = zeros(1,H+1);    % 初始化Δ_wo=0
    for i=1:N
        %% 前向传播
        % 隐藏层，激活函数sigmoid
        ho = sigmoid( X(i,:) * W' ) ;   % 隐藏层输出结果
        h_t = [1,ho ];                  % 隐藏层输出结果增广向量
        yo = h_t * wo';   
        %% 后向传播
         % 计算输出层δ
        Err_wo = ( t(i)-yo );          
        delta_wo = delta_wo + [1, ho ] * Err_wo;       
        % 计算隐藏层层δ
        Err_W = ho .* ( 1- ho ) * Err_wo .* wo(2:H+1);  
        delta_W = delta_W + repmat( Err_W',1,n+1) .* repmat(X(i,:),H,1) ;
    end      
    %% 更新权重
    wo = wo + step*delta_wo;
    W = W + step.*delta_W ;   
    cost = 0;
    %% 计算training loss（MSE）
    for i=1:N
        ho = sigmoid( X(i,:) * W' ) ;    
        yo = [1, ho ] * wo';
        cost = cost + (yo - t(i) ) ^2;
    end      
    iter = iter  + 1;
    cost_training(iter) = cost / N;      
    %% 计算validation loss（MSE）
    predicts = BPNN_predicts( v_X, W, wo );
    e = predicts - v_truth ;
    cost_val(iter) = e' * e / v_N ;
    
end
output.W = W;
output.wo = wo;
output.cost = cost_training;
output.cost_val = cost_val;
% End function BPNN()

%% 根据权重，计算输出层预测结果
function predicts = BPNN_predicts( X, W, wo )
[N,~] = size(X);
X = [ ones(N,1) , X ] ;
predicts = zeros(N,1);
for i=1:N
        ho = sigmoid( X(i,:) * W' ) ;
        h_t = [1,ho ];
        predicts(i) =  ceil(h_t * wo');       
        if predicts(i) < 0 
            predicts(i) = 1 ;
        end   
end      
% End function BPNN_predicts()


