function predicts = use_tree_predict(data, matcheds, tree, U_label)  
%Classify recursively using a tree  
% data -- ����������������ݾ���
% matcheds -- һ������
% tree -- �����ľ�����
% % U_label -- ���ظ����ֵı�ǩ�� [-1,+1]
[N_sample, N_attri]=size(data);
predicts = zeros(N_sample,1); %����ÿ�������ĳ�ʼԤ���ǩ����0 
% �����ı߽�����
if (tree.dim == 0)  %��˵���ﵽ������Ҷ�ӽڵ�
    predicts(matcheds) = tree.child;  %�õ�������Ӧ�ı�ǩ��tree.child
    return  
end  

%% �������սڵ㣬����������
dim = tree.dim;  %�õ���������
dims= 1:N_attri;  %�õ���������
dims(dims==dim)=[]; % ɾ���������

%% ���ݵõ��ľ������Բ����������з���
U_Avalue = unique(data( : ,dim)); %�õ������������������������ظ�����ֵ�ļ���
for i = 1:length(U_Avalue)  %����ÿ������ֵ  ,��������ֵ����һ����
    % ���U_Avalue(i)����tree.Nf�г��ֹ�������0
    if any( tree.Nf==U_Avalue(i) )  
    %tree.NfΪ���ķ����������� ��ǰ�����������������������ֵ
        data0 = data( : , dims);
        matcheds0   = matcheds(find(data(matcheds,dim) == U_Avalue(i)));  
        %�ҵ���ǰ�����������������������ֵ==����ֵ����������
        tree0 = tree.child(find(U_Avalue(i)==tree.Nf));
        predicts = predicts + use_tree_predict(data0, matcheds0, tree0, U_label);
        %���ⲿ�������ٷֲ� 
    else % ����ֵû�г��ֹ�
        data0 = data( : , dims);
        matcheds0  = matcheds(find(data(matcheds,dim) == U_Avalue(i))); 
        tree0 = tree.child( length(tree.Nf) );  % ��������Ľ��
        predicts = predicts + use_tree_predict(data0, matcheds0, tree0, U_label);
        %���ⲿ�������ٷֲ� 
    end  
end  
  
%END use_tree  