/*
注意修改文件路径
test_set.csv需要手动删除第一列 
*/ 
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>
#define SIZE 1000000 // 文章字符上限  
#define WORD_SIZE 500000 // 文章字符上限 
#define PASSAGE_NUM 1000	// 训练集文章上限 
#define WORD_NUM 60000	// 单词上限 

using namespace std;
using namespace boost;

class Passage
{
public: 
	string label ;   		//每一篇文章的label 
	vector<string> voc ;	//每一篇文章的单词库 
	
	Passage()
	{
		label="" ;
	}
				
} ;

class Pos
{
public:
	int row, col;
	Pos()
	{
		row = 0;
		col = 0;
	}
	Pos( const int r, const int c ) 
	{
		row = r;
		col = c;
	}	
	//按照顺序排列 
	bool operator<( const Pos& p2) const
	{
		if( this->row != p2.row )
			return this->row < p2.row ;
		else // this->row == t2.row 
		{
			if( this->col==p2.col ) return false ;
			else return this->col<p2.col ; //从小到大排序
		}
	}
};

vector<string> word;    //存training单词 
vector<Passage> training ; //训练集的所有文章 
vector<Passage> newdata ;  //测试集的所有文章 
map<Pos, double> TF0, TF1;  //辅助计算TF矩阵的map

int OneHot_training[PASSAGE_NUM][WORD_NUM] ; //训练集OneHot_training矩阵 
double TF_training[PASSAGE_NUM][WORD_NUM] ; //训练集TF_training矩阵 
int OneHot_new[PASSAGE_NUM][WORD_NUM] ; //新的数据集(验证集/测试集)OneHot 矩阵 
double TF_new[PASSAGE_NUM][WORD_NUM] ; //新的数据集(验证集/测试集)OneHot 矩阵 
// PASSAGE_NUM 3000训练集文章上限 ,WORD_NUM 60000单词上限 

int main()
{
	/** 打开training文件，读取文件，分割为文本和标签 **/ 	 
	ifstream file0;   
	file0.open("D:\\文档\\MATLAB\\BPNN_train_small.csv");
	char *str0;	
	str0 = new char[SIZE] ;	
	file0.get(str0,SIZE,'\0');     //从文件中读取字符到字符串str1，当遇到字符'\0'或读取了SIZE个字符时终止。
	string strTmp0 = str0;
	file0.close();	
	vector<string> SplitVec0;
	split(SplitVec0, strTmp0, is_any_of(",\n"), token_compress_on ); //分割训练集文本字符串
	vector<string>::iterator begin = SplitVec0.begin() ;
	SplitVec0.erase(begin) ;
	SplitVec0.erase(begin) ;
	
	int num_training = SplitVec0.size() / 2 ;	//文章数量 	
	/** FINISHED 打开training文件，读取文件，分割为文本和标签 **/
	
	/* 构造训练集OneHot矩阵和TF矩阵 */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//构造第i篇文章  	 
		p.label = SplitVec0[2*i+1] ; 	//每一篇文章的label 
		string tmp = SplitVec0[2*i] ;       	//第i篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第i篇文章的单词
		training.push_back( p ) ;		//第i篇文章插入training文章vector 

		//构造第i行的OneHot和TF矩阵 
		for( int j=0; j<p.voc.size(); j++ ) // 查看文章中每个单词 
		{
			string subtmp = p.voc[j] ;	   // 第i篇文章的第j个单词 
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //单词表已经存在这个单词 
			{
				int loc = iterloc-word.begin() ;  	// 第j个单词所在单词库位置 
				OneHot_training[i][loc] = 1;
				int oldVal = TF0[ Pos(i, loc )] ; //矩阵位置的员原来的值  
				TF0[ Pos(i, loc )] = oldVal+1 ;  //单词再一次出现，修改map中这个矩阵位置的值
			}
			else //新单词 
			{	
				word.push_back( subtmp ) ;			// 插入新单词 
				OneHot_training[i][ word.size()-1 ] = 1; 	// 第j个单词所在单词库位置 
				TF0[ Pos(i, word.size()-1 ) ] = 1 ;	  //新词第一次出现
			}		
		}				
	}
	
	/* FINISHED 构造training文章vector, OneHot_training矩阵 */ 
	
	/* 稀疏矩阵TF转成二维数组 */
	//出现次数->TF 
	map<Pos, double>::iterator it = TF0.begin();  
	map<Pos, double>::iterator end = TF0.end();
	for( ; it!=end; it++)
	{	// TF= 词语出现的次数/文章总单词数  
		it->second = it->second / training[it->first.row].voc.size() ;
	}	
	it = TF0.begin();
	for( ; it!=end; it++ ) 
	{
		TF_training[it->first.row][it->first.col] = it->second ;		
	}
	/* FINISHED 稀疏矩阵TF转成二维数组 */
	
	/** 打开validation文件，读取文件，分割为文本和标签 **/ 	   
	ifstream file1; 
	file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\分类原始数据\\test_set.csv");
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //从文件中读取字符到字符串str1，当遇到字符'\0'或读取了SIZE个字符时终止。
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //分割字符串 以Tab和换行为分隔符
	begin = SplitVec1.begin() ;
	SplitVec1.erase(begin) ;
	SplitVec1.erase(begin) ;
	int num_new = SplitVec1.size() / 2 ;	//validation文章数量 	
	/** FINISHED 打开validation文件，读取文件，分割为文本和标签 **/

	/* 构造validation文章OneHot_training矩阵 */ 
	
	for (int i = 0; i<num_new; i++)	//第i篇文章 
	{
		Passage p;      				//构造第i篇文章  	 
		p.label = SplitVec1[2*i+1] ; 	//每一篇文章的label 
		string tmp = SplitVec1[2*i] ;       	//第i篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第i篇文章的单词
		newdata.push_back( p ) ;		//第i篇文章插入training文章vector 
		
		for( int j=0; j<p.voc.size(); j++ ) //第j个单词 
		{ // 查看文章中每个单词 
			string subtmp = p.voc[j] ;	   // 第i篇文章的第j个单词 
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //单词表已经存在这个单词 
			{
				int loc = iterloc-word.begin() ;  	// 第j个单词所在单词库位置 
				OneHot_new[i][loc] = 1;
				int oldVal = TF1[ Pos(i, loc )] ; //矩阵位置的员原来的值  
				TF1[ Pos(i, loc )] = oldVal+1 ;  //单词再一次出现，修改map中这个矩阵位置的值
			}
			else //新单词 
			{	
				word.push_back( subtmp ) ;			// 插入新单词 
				OneHot_new[i][ word.size()-1 ] = 1; 	// 第j个单词所在单词库位置 
				TF1[ Pos(i, word.size()-1 ) ] = 1 ;	   // 频次矩阵 
			}		
		}
						
	}
	/* FINISHED 构造training文章vector, OneHot_training矩阵 */ 
	
	/* 稀疏矩阵TF转成二维数组 */
	//出现次数->TF 
	map<Pos, double>::iterator it1 = TF1.begin();  
	map<Pos, double>::iterator end1 = TF1.end();
	int times ;
	for( ; it1!=end1; it1++)
	{	// TF= 词语出现的次数/文章总单词数  
		times = newdata[it1->first.row].voc.size() ;
		it1->second = it1->second / times;
	}		
	it1 = TF1.begin();
	for( ; it1!=end1; it1++ ) 
	{
		TF_new[it1->first.row][it1->first.col] = it1->second ;		
	}
	/* 稀疏矩阵TF转成二维数组 */

	/* KNN 计算 */ 
	ofstream ofile_new;               //定义输出文件OneHot.txt
	ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\分类原始数据\\15352204_LinDan_KNN_classification.csv");    //作为输出文件打开	
	ofile_new <<"textid" <<"," <<"label" <<endl ;
	
	double max_rate = 0;    // 对不同的K,计算准确率最大值 
	int max_rate_k = 0;     // 准确率最大值对应的K 
	for( int k=14; k<=14; k++ )    //KNN算法不同K值 
	{
		int num_correct = 0;		//测试结果准确的个数 
		for( int r=0; r<num_new; r++)  //新的数据集每一篇文章，第r篇文章 
		{
			string answer = SplitVec1[2*r+1] ; 	//新集第r篇文章的label ，正确答案 
			multimap<double,string,greater<double>> knn ;  
			//对测试集每一篇文章有一个knn的排序，前k个距离最近的文章（距离，标签）对 
			
			for( int c=0; c<num_training; c++) //对训练集的遍历 
			{
				double dis=0 ;			// 新集第r篇与训练集第c篇 
				double len1 = 0, len2 = 0, dot_product = 0 ;
				for( int j=0; j<word.size(); j++) 
				{	// 欧式距离 
					// dis += ( TF_training[c][j] - TF_new[r][j] )*( TF_training[c][j] - TF_new[r][j] ) ;	
					len1 += OneHot_training[c][j] ;
					len2 += OneHot_new[r][j] ;
					dot_product += OneHot_training[c][j] * OneHot_new[r][j] ;								
				}
				dis= dot_product / sqrt( len1 * len2 ) ;
				knn.insert( multimap<double,string>::value_type(dis,SplitVec0[2*c+1])) ;
				// 插入测试集第r篇与训练集第c篇 （距离，标签），自动按照距离排序 
			}					
			
			int count[6] ;
			for(int i=0; i<6; i++)
				count[i] = 0;   
			// 对距离最近的K个训练集文章进行“多数投票” 			
			string label[6]={ "anger", "disgust", "fear", "joy", "sad", "surprise" } ;
			multimap<double,string>::iterator knnit = knn.begin();			
			for( int i=0; i<k; i++, knnit++)
			{
				string tmp = knnit->second ;
				// 对前K个训练集文章的标签计数 
				if( tmp=="anger") 			count[0]++ ;
				else if( tmp=="disgust") 	count[1]++ ;
				else if( tmp=="fear") 		count[2]++ ;
				else if( tmp=="joy") 		count[3]++ ;
				else if( tmp=="sad") 		count[4]++ ;
				else if( tmp=="surprise") 	count[5]++ ;
				else cout<<endl<<"error:" <<knnit->second <<endl ;
			}
			int maxNum = 0;		//前k名出现的标签次数的最大值   
			int max_i = 0; 		//最大值对应的标签下标 
			//找出前k名出现的标签次数的最大值  
			for( int i=0; i<6; i++ )
			{
				if ( count[i]>maxNum )
				{
					maxNum = count[i] ;
					max_i = i;
				} 
			}
			string predict = label[max_i] ; 	
			//根据最大值对应的标签下标，找到预测的标签结果 		
			ofile_new <<r+1 <<"," <<predict <<endl ; 
			if( predict == answer ) // 验证 
				num_correct ++ ;    // 计算准确的个数 
		}
		double rate = num_correct*100.0 / num_new ;  // 计算准确率 
		// printf("%d num_correct=%d  rate= %f\n", k,num_correct, rate );
		// 计算不同K值中，使得准确率最大的K值 
		if( rate > max_rate )
		{
			max_rate = rate ;   // 最大准确率 
			max_rate_k = k ;    // 准确率最大的K值 
		}
	}
	// cout <<" **max_k" <<max_rate_k <<"  max_rate " <<max_rate <<endl ;	
	ofile_new.close() ;
	/* FINISHED KNN 分类计算 */
}

