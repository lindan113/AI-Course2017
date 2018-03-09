function predicts = use_tree_predict(data, matcheds, tree, U_label)  
%Classify recursively using a tree  
% data -- 输入的样本特征数据矩阵
% matcheds -- 一个向量
% tree -- 建立的决策树
% % U_label -- 不重复出现的标签， [-1,+1]
[N_sample, N_attri]=size(data);
predicts = zeros(N_sample,1); %设置每个样本的初始预测标签都是0 
% 迭代的边界条件
if (tree.dim == 0)  %这说明达到了树的叶子节点
    predicts(matcheds) = tree.child;  %得到样本对应的标签是tree.child
    return  
end  

%% 不是最终节点，接续往下找
dim = tree.dim;  %得到分裂特征
dims= 1:N_attri;  %得到特征索引
dims(dims==dim)=[]; % 删除这个特征

%% 根据得到的决策树对测试样本进行分类
U_Avalue = unique(data( : ,dim)); %得到这个样本集中这个特征的无重复特征值的集合
for i = 1:length(U_Avalue)  %遍历每个特征值  ,根据特征值来建一颗树
    % 如果U_Avalue(i)不在tree.Nf中出现过，返回0
    if any( tree.Nf==U_Avalue(i) )  
    %tree.Nf为树的分类特征向量 当前所有样本的这个特征的特征值
        data0 = data( : , dims);
        matcheds0   = matcheds(find(data(matcheds,dim) == U_Avalue(i)));  
        %找到当前测试样本中这个特征的特征值==分裂值的样本索引
        tree0 = tree.child(find(U_Avalue(i)==tree.Nf));
        predicts = predicts + use_tree_predict(data0, matcheds0, tree0, U_label);
        %对这部分样本再分叉 
    else % 特征值没有出现过
        data0 = data( : , dims);
        matcheds0  = matcheds(find(data(matcheds,dim) == U_Avalue(i))); 
        tree0 = tree.child( length(tree.Nf) );  % 例外情况的结点
        predicts = predicts + use_tree_predict(data0, matcheds0, tree0, U_label);
        %对这部分样本再分叉 
    end  
end  
  
%END use_tree  