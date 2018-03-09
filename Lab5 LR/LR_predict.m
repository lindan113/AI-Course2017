function test_predicts = LR_predict(testVectors, W)
[testRow,~] = size(testVectors);       % ��ȡ �������������������ĳ���
testVectors= sigmod( testVectors ) ;    % ��׼��

x_ = [ ones(testRow,1),testVectors ] ; % ��ӵ�һ��1
p_ = sigmod( x_ * W );
test_predicts = zeros(testRow,1);
test_predicts( find( p_ >= 0.5 ) ) = 1;  % ���ڵ���0.5����Ϊ1  ����
