function W = LR_makeW(trainVectors, limit, alpha)
[trainRow,trainColumn] = size(trainVectors);  % ��ȡ [�������������������ĳ���+1]
Y = trainVectors( : ,trainColumn) ;   % 1������  0:δ����
data = trainVectors( : ,1:trainColumn-1 ) ;

data = sigmod( data );% ��׼��
X = [ ones(trainRow,1), data ] ; % ��ӵ�һ��1

% ��ʼ��Ȩ������w 
 W = ones(trainColumn,1);  

Iteration = 1 ;
while(true)
    A = X * W ; % ���㡰������
    L = Y .* A - log(1+exp(A)) ; % ������Ȼ�� ��������1
    J = sum(L)/trainRow;  % ���ۺ���
    grad = X' * (sigmod(A) - Y);
    if Iteration >= limit ||sum(grad) == 0.0000001
        break ;
    end
     W = W - alpha * grad ;
    Iteration = Iteration + 1 ;
end 

