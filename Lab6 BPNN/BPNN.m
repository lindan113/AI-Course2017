
function output = BPNN(X, t, v_X, v_truth,H, max_iter,step)
% X -- ����ѵ�����������������ݾ���N��,��N��������n�У���n��������
% t -- ����ѵ�����ݵı�ǩ��N��1
% v_X -- ������֤���������������ݾ���
% v_truth -- ������֤���ݵı�ǩ
% H -- ���ز��

[N,n] = size(X);
% N -- ѵ��������������
% n -- ѵ����������������
v_N = size(v_X, 1) ;
% N -- ��֤��������������
X = [ ones(N,1) , X ] ;
% �������

if N~= length(t)
    error('inconsistent sample size');
end

%% �����ʼ�� 
W = rand(H,n+1);        % W --  ���뵽���ز��Ȩ�أ�����ƫ�ã�
wo = rand(1,H+1);       % wo -- ���ز㵽������Ȩ�أ�����ƫ�ã�  
% W = randn(H,n+1);        % W --  ���뵽���ز��Ȩ�أ�����ƫ�ã�
% wo = randn(1,H+1);       % wo -- ���ز㵽������Ȩ�أ�����ƫ�ã�  
iter = 0;       % ��������
cost_training = zeros(1,max_iter);  % training loss
cost_val = zeros(1,max_iter);       % validation loss
while iter < max_iter       
    delta_W = zeros(H,n+1);     % ��ʼ����_W=0
    delta_wo = zeros(1,H+1);    % ��ʼ����_wo=0
    for i=1:N
        %% ǰ�򴫲�
        % ���ز㣬�����sigmoid
        ho = sigmoid( X(i,:) * W' ) ;   % ���ز�������
        h_t = [1,ho ];                  % ���ز���������������
        yo = h_t * wo';   
        %% ���򴫲�
         % ����������
        Err_wo = ( t(i)-yo );          
        delta_wo = delta_wo + [1, ho ] * Err_wo;       
        % �������ز���
        Err_W = ho .* ( 1- ho ) * Err_wo .* wo(2:H+1);  
        delta_W = delta_W + repmat( Err_W',1,n+1) .* repmat(X(i,:),H,1) ;
    end      
    %% ����Ȩ��
    wo = wo + step*delta_wo;
    W = W + step.*delta_W ;   
    cost = 0;
    %% ����training loss��MSE��
    for i=1:N
        ho = sigmoid( X(i,:) * W' ) ;    
        yo = [1, ho ] * wo';
        cost = cost + (yo - t(i) ) ^2;
    end      
    iter = iter  + 1;
    cost_training(iter) = cost / N;      
    %% ����validation loss��MSE��
    predicts = BPNN_predicts( v_X, W, wo );
    e = predicts - v_truth ;
    cost_val(iter) = e' * e / v_N ;
    
end
output.W = W;
output.wo = wo;
output.cost = cost_training;
output.cost_val = cost_val;
% End function BPNN()

%% ����Ȩ�أ����������Ԥ����
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


