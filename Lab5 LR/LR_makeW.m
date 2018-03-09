function W = LR_makeW(trainVectors, limit, alpha)
[trainRow,trainColumn] = size(trainVectors);  % 获取 [样本个数，特征向量的长度+1]
Y = trainVectors( : ,trainColumn) ;   % 1：发生  0:未发生
data = trainVectors( : ,1:trainColumn-1 ) ;

data = sigmod( data );% 标准化
X = [ ones(trainRow,1), data ] ; % 添加第一个1

% 初始化权重向量w 
 W = ones(trainColumn,1);  

Iteration = 1 ;
while(true)
    A = X * W ; % 计算“分数”
    L = Y .* A - log(1+exp(A)) ; % 对数似然度 样本数×1
    J = sum(L)/trainRow;  % 代价函数
    grad = X' * (sigmod(A) - Y);
    if Iteration >= limit ||sum(grad) == 0.0000001
        break ;
    end
     W = W - alpha * grad ;
    Iteration = Iteration + 1 ;
end 

