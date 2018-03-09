function tree = make_tree(data, labels, inc_node, mode) 
% data -- 输入的样本特征数据矩阵，行为样本，列为特征， N_sample*N_attri
% labels -- 输入的样本标签向量 
% inc_node -- 递归边界，数据子集的样本个数
% mode -- 模式，1: ID3;  2:C4.5;  3:GINI

%% 函数部分：
[N_sample, N_attri]=size(data);
U_label=unique(labels);     % U_label = [-1,+1]
N_Ulabel = length(U_label);  % N_label = 2
tree.dim = 0;
%tree.child = [];

%% 如果剩余训练样本太小(小于inc_node)，或只剩一个，或只剩一类标签，退出  
if ((inc_node > N_attri) | (N_attri == 1) | (N_Ulabel == 1)) 
    %如果剩余训练样本太小(小于inc_node)，或只剩一个，或只剩一类标签，退出  
    H = hist(labels, N_Ulabel);  %统计样本的标签，分别属于每个标签的数目 H(1×标签数目)
    [~,largest]     = max(H); 
    %得到包含样本数最多的那个标签的索引，记为largest 包含多少个样本，记为m
    tree.Nf         = [];  
    tree.child      = U_label(largest);%姑且直接返回其中包含样本数最多一类作为其标签 
    return  
end  

%% 计算数据集的信息熵 H_Data
%P_label = zeros(1,N_Ulabel);
for i=1:N_Ulabel
    P_label(i) = length( find( labels == U_label(i))) / N_sample;
end
H_Data = sum( -P_label .* H_log( P_label ) );

%% 计算每个attribute的增益&信息增益率
gain = zeros(1,N_attri);        % 增益
gainRatio = zeros(1,N_attri);   % 增益率
GINI = zeros(1,N_attri);        % GINI指数
for x = 1:N_attri %遍历每一个特征
    S_Avalue = data( : , x);        % 每一个特征的所有样本聚合
    U_Avalue = unique( S_Avalue );  % 该特征的不重复特征值集合
    N_UAvalue = length( U_Avalue ); % 该特征的不重复特征值个数 
    P_Matrix = zeros( N_Ulabel, N_UAvalue); % 特征矩阵（概率矩阵）的构造
    for i=1:N_Ulabel % 两行 -1, +1
        for j=1:N_UAvalue  
            matcheds = find( (labels==U_label(i))&(S_Avalue==U_Avalue(j)) ); 
            P_Matrix(i,j) = length(matcheds);
        end      
    end
    Pk = sum(P_Matrix);  
    %取P的每一列的和，也就是得到当前所有样本中，这个特征的特征值==这个特征值的样本数目   
    %Pk(1×特征值数目)表示这个特征的特征值等于每个特征值的样本数目
    P_sum = repmat(Pk, N_Ulabel, 1);  %把Pk复制成判别标签个数 的行, 列还是一组
    %这主要在保证P1作被除数时不等于0
    P_sum = P_sum + eps*(P_sum==0);  
    P_Matrix = P_Matrix ./ P_sum;  
    % 得到当前所有样本中，这个特征的值等于每个特征值且标签等于每个标签的样本，
    % 占当前这个特征值中的样本的比例
    Pk = Pk / sum(Pk);  
    %得到当前所有样本中，这个特征的值等于每个特征值的样本，
    % 占当前样本总数的比例
    % 例如 A=age: Pk = [5/14, 4/14, 5/14];

    %% 计算 条件概率
    % Sum_subH = sum(-P_Matrix .* log2(P_Matrix)); 
    Sum_subH = sum(-P_Matrix .* H_log(P_Matrix));  
    %对特征集中每个特征分别计算信息熵  info(1×特征值数目)   
    H_DA = sum( Pk .* Sum_subH );
    %% 计算gain和gainRatio
    gain(x) = H_Data - H_DA;              % 信息增益
    splitInfo = (sum(-Pk .* H_log(Pk)));  % 特征本身的不确定性
    gainRatio(x) = gain(x)/ splitInfo ;   % 信息增益率
    %% 计算GINGI指数
    sub_gini = sum( P_Matrix .* ( 1-P_Matrix));
    GINI(x) = sum( Pk .* sub_gini ) ;
end

%% 选出分裂的特征
if mode == 1
    [~,dim] = max(gain);
elseif mode == 2
    [~,dim] = max(gainRatio);
elseif mode ==3
    [~,dim] = min(GINI);
end
tree.dim = dim ;%记为树的分裂特征
Nf = unique(data(:,dim));   
%得到选择的这个作为分裂特征的特征的那一行  也就是得到当前所有样本的这个特征的特征值
Nbins = length(Nf);         %得到这个特征的无重复的特征值的数目
tree.Nf = Nf;               %记为树的分类特征向量 当前所有样本的这个特征的特征值

%% 判断是否继续从数据子集分裂
H  = hist(labels,N_Ulabel );  
%统计当前所有样本的标签，分别属于每个标签的数目    H(1×标签数目)
[~,largest] = max(H);  
%得到包含样本数最多的那个标签的索引，记为largest 包含多少个样本，记为m
if Nbins == 1  %无重复的特征值的数目==1，即这个特征只有这一个特征值，就不能进行分裂
    tree.Nf = [];  %因为不以这个特征进行分裂，所以Nf和split_loc都为空
    tree.child= U_label(largest);  %姑且将这个特征的标签就记为包含样本数最多的那个标签
    return  
end  

%% 增加一个多余的叶子节点，处理测试集中没有出现过的特征值
tree.child(Nbins+1).dim = 0;
tree.child(Nbins+1).Nf = [];
tree.child(Nbins+1).child = U_label(largest);

%% 创建子节点
dims = 1:N_attri ;
dims(dims==dim)=[]; % 删除这个特征
%如果当前选择的这个作为分裂特征的特征是个离散特征 
for i = 1:Nbins   %遍历这个特征下无重复的特征值的数目
    matcheds = find(data(:,dim) == Nf(i));  
    %找到当前所有样本的这个特征的特征值为Nf(i)的索引们
    data0 = data(matcheds,dims);
    label0 = labels(matcheds);
    tree.child(i)   = make_tree( data0, label0,inc_node, mode);%递归
    %因为这是个离散特征，所以分叉成Nbins个，分别针对每个特征值里的样本，进行再分叉
end  


