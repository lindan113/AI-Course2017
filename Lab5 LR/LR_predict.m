function test_predicts = LR_predict(testVectors, W)
[testRow,~] = size(testVectors);       % 获取 样本个数×特征向量的长度
testVectors= sigmod( testVectors ) ;    % 标准化

x_ = [ ones(testRow,1),testVectors ] ; % 添加第一个1
p_ = sigmod( x_ * W );
test_predicts = zeros(testRow,1);
test_predicts( find( p_ >= 0.5 ) ) = 1;  % 大于等于0.5的判为1  发生
