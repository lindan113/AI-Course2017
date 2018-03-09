function tree = make_tree(data, labels, inc_node, mode) 
% data -- ����������������ݾ�����Ϊ��������Ϊ������ N_sample*N_attri
% labels -- �����������ǩ���� 
% inc_node -- �ݹ�߽磬�����Ӽ�����������
% mode -- ģʽ��1: ID3;  2:C4.5;  3:GINI

%% �������֣�
[N_sample, N_attri]=size(data);
U_label=unique(labels);     % U_label = [-1,+1]
N_Ulabel = length(U_label);  % N_label = 2
tree.dim = 0;
%tree.child = [];

%% ���ʣ��ѵ������̫С(С��inc_node)����ֻʣһ������ֻʣһ���ǩ���˳�  
if ((inc_node > N_attri) | (N_attri == 1) | (N_Ulabel == 1)) 
    %���ʣ��ѵ������̫С(С��inc_node)����ֻʣһ������ֻʣһ���ǩ���˳�  
    H = hist(labels, N_Ulabel);  %ͳ�������ı�ǩ���ֱ�����ÿ����ǩ����Ŀ H(1����ǩ��Ŀ)
    [~,largest]     = max(H); 
    %�õ����������������Ǹ���ǩ����������Ϊlargest �������ٸ���������Ϊm
    tree.Nf         = [];  
    tree.child      = U_label(largest);%����ֱ�ӷ������а������������һ����Ϊ���ǩ 
    return  
end  

%% �������ݼ�����Ϣ�� H_Data
%P_label = zeros(1,N_Ulabel);
for i=1:N_Ulabel
    P_label(i) = length( find( labels == U_label(i))) / N_sample;
end
H_Data = sum( -P_label .* H_log( P_label ) );

%% ����ÿ��attribute������&��Ϣ������
gain = zeros(1,N_attri);        % ����
gainRatio = zeros(1,N_attri);   % ������
GINI = zeros(1,N_attri);        % GINIָ��
for x = 1:N_attri %����ÿһ������
    S_Avalue = data( : , x);        % ÿһ�����������������ۺ�
    U_Avalue = unique( S_Avalue );  % �������Ĳ��ظ�����ֵ����
    N_UAvalue = length( U_Avalue ); % �������Ĳ��ظ�����ֵ���� 
    P_Matrix = zeros( N_Ulabel, N_UAvalue); % �������󣨸��ʾ��󣩵Ĺ���
    for i=1:N_Ulabel % ���� -1, +1
        for j=1:N_UAvalue  
            matcheds = find( (labels==U_label(i))&(S_Avalue==U_Avalue(j)) ); 
            P_Matrix(i,j) = length(matcheds);
        end      
    end
    Pk = sum(P_Matrix);  
    %ȡP��ÿһ�еĺͣ�Ҳ���ǵõ���ǰ���������У��������������ֵ==�������ֵ��������Ŀ   
    %Pk(1������ֵ��Ŀ)��ʾ�������������ֵ����ÿ������ֵ��������Ŀ
    P_sum = repmat(Pk, N_Ulabel, 1);  %��Pk���Ƴ��б��ǩ���� ����, �л���һ��
    %����Ҫ�ڱ�֤P1��������ʱ������0
    P_sum = P_sum + eps*(P_sum==0);  
    P_Matrix = P_Matrix ./ P_sum;  
    % �õ���ǰ���������У����������ֵ����ÿ������ֵ�ұ�ǩ����ÿ����ǩ��������
    % ռ��ǰ�������ֵ�е������ı���
    Pk = Pk / sum(Pk);  
    %�õ���ǰ���������У����������ֵ����ÿ������ֵ��������
    % ռ��ǰ���������ı���
    % ���� A=age: Pk = [5/14, 4/14, 5/14];

    %% ���� ��������
    % Sum_subH = sum(-P_Matrix .* log2(P_Matrix)); 
    Sum_subH = sum(-P_Matrix .* H_log(P_Matrix));  
    %����������ÿ�������ֱ������Ϣ��  info(1������ֵ��Ŀ)   
    H_DA = sum( Pk .* Sum_subH );
    %% ����gain��gainRatio
    gain(x) = H_Data - H_DA;              % ��Ϣ����
    splitInfo = (sum(-Pk .* H_log(Pk)));  % ��������Ĳ�ȷ����
    gainRatio(x) = gain(x)/ splitInfo ;   % ��Ϣ������
    %% ����GINGIָ��
    sub_gini = sum( P_Matrix .* ( 1-P_Matrix));
    GINI(x) = sum( Pk .* sub_gini ) ;
end

%% ѡ�����ѵ�����
if mode == 1
    [~,dim] = max(gain);
elseif mode == 2
    [~,dim] = max(gainRatio);
elseif mode ==3
    [~,dim] = min(GINI);
end
tree.dim = dim ;%��Ϊ���ķ�������
Nf = unique(data(:,dim));   
%�õ�ѡ��������Ϊ������������������һ��  Ҳ���ǵõ���ǰ�����������������������ֵ
Nbins = length(Nf);         %�õ�������������ظ�������ֵ����Ŀ
tree.Nf = Nf;               %��Ϊ���ķ����������� ��ǰ�����������������������ֵ

%% �ж��Ƿ�����������Ӽ�����
H  = hist(labels,N_Ulabel );  
%ͳ�Ƶ�ǰ���������ı�ǩ���ֱ�����ÿ����ǩ����Ŀ    H(1����ǩ��Ŀ)
[~,largest] = max(H);  
%�õ����������������Ǹ���ǩ����������Ϊlargest �������ٸ���������Ϊm
if Nbins == 1  %���ظ�������ֵ����Ŀ==1�����������ֻ����һ������ֵ���Ͳ��ܽ��з���
    tree.Nf = [];  %��Ϊ��������������з��ѣ�����Nf��split_loc��Ϊ��
    tree.child= U_label(largest);  %���ҽ���������ı�ǩ�ͼ�Ϊ���������������Ǹ���ǩ
    return  
end  

%% ����һ�������Ҷ�ӽڵ㣬������Լ���û�г��ֹ�������ֵ
tree.child(Nbins+1).dim = 0;
tree.child(Nbins+1).Nf = [];
tree.child(Nbins+1).child = U_label(largest);

%% �����ӽڵ�
dims = 1:N_attri ;
dims(dims==dim)=[]; % ɾ���������
%�����ǰѡ��������Ϊ���������������Ǹ���ɢ���� 
for i = 1:Nbins   %����������������ظ�������ֵ����Ŀ
    matcheds = find(data(:,dim) == Nf(i));  
    %�ҵ���ǰ�����������������������ֵΪNf(i)��������
    data0 = data(matcheds,dims);
    label0 = labels(matcheds);
    tree.child(i)   = make_tree( data0, label0,inc_node, mode);%�ݹ�
    %��Ϊ���Ǹ���ɢ���������Էֲ��Nbins�����ֱ����ÿ������ֵ��������������ٷֲ�
end  


