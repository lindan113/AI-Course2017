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
	// anger,disgust,fear,joy,sad,surprise
	double pro[6] ;   		//每一篇文章的label 
	vector<string> voc ;	//每一篇文章的单词库 
	
	Passage()
	{
		for( int i=0; i<6; i++)
		{
			pro[i]=0;
		} 
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
	file0.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\train_set.csv");
	char *str0;	
	str0 = new char[SIZE] ;	
	file0.get(str0,SIZE,'\0');     //从文件中读取字符到字符串str1，当遇到字符'\0'或读取了SIZE个字符时终止。
	string strTmp0 = str0;
	file0.close();	
	vector<string> SplitVec0;
	split(SplitVec0, strTmp0, is_any_of(",\n"), token_compress_on ); //分割训练集文本字符串
	vector<string>::iterator begin = SplitVec0.begin() ;
	for( int i=0; i<7; i++ ) //有7个要删掉 
		SplitVec0.erase(begin) ;		// 删除第一行的七项 
	
	int num_training = SplitVec0.size() / 7 ;	//文章数量 	
	/** FINISHED 打开training文件，读取文件，分割为文本和标签 **/

	
	/* 构造训练集OneHot矩阵和TF矩阵 */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//构造第i篇文章  	 
		for( int u=0; u<=5; u++)
		{	// string类型转double类型		
			p.pro[u]= std::stod( SplitVec0[7*i+1+u] ) ;	 //第i篇文章的情感概率		
		}	
		string tmp = SplitVec0[7*i] ;       	//第i篇文章没有截断前 	
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
	file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\test_set.csv");
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //从文件中读取字符到字符串str1，当遇到字符'\0'或读取了SIZE个字符时终止。
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //分割字符串 以Tab和换行为分隔符
	begin = SplitVec1.begin() ;
	for( int i=0; i<7; i++ )
		SplitVec1.erase(begin) ;		// 删除第一行的七项 

	int num_new = SplitVec1.size() / 7 ;	//validation文章数量 	
	/** FINISHED 打开validation文件，读取文件，分割为文本和标签 **/

	/* 构造validation文章OneHot_training矩阵 */ 
	
	for (int i = 0; i<num_new; i++)	//第i篇文章 
	{
		Passage p;      				//构造第i篇文章  	 
		string tmp = SplitVec1[7*i] ;       	//第i篇文章没有截断前 		
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
				//OneHot0[ Pos(i, loc ) ] = 1 ;  //出现为1 	
				OneHot_new[i][loc] = 1;
				int oldVal = TF1[ Pos(i, loc )] ; //矩阵位置的员原来的值  
				TF1[ Pos(i, loc )] = oldVal+1 ;  //单词再一次出现，修改map中这个矩阵位置的值
			}
			else //新单词 
			{	
				word.push_back( subtmp ) ;			// 插入新单词 
				OneHot_new[i][ word.size()-1 ] = 1; 	// 第j个单词所在单词库位置 
				//OneHot0[ Pos(i, word.size()-1 ) ] = 1 ;  // OneHot 矩阵 
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
	/* * * * * * * * * * * * * * * 大数据 * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //定义输出文件OneHot.txt
	ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\15352204_LinDan_KNN_regression.csv");    //作为输出文件打开	
	//ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nb_reg_small\\validation相关度评估.xlsx");   
	ofile_new <<"id" <<"," <<"anger" <<"," <<"disgust" <<"," <<"fear"
					<<"," <<"joy" <<"," <<"sad" <<"," <<"surprise" <<endl ;	
	
	double pro_emo[6] ; // 每篇文章6种情感概率 
	for( int k=15; k==15; k++ )    //KNN算法不同K值 
	{
		for( int r=0; r<num_new; r++)  //新的数据集每一篇文章，第r篇文章 
		{						
			//multimap<double,int,greater<double>> knn ; //余弦距离 
			multimap<double,int,less<double>> knn ;     //欧式距离
			//对测试集每一篇文章有一个knn的排序，前k个距离最近的文章（距离，标签）
			for( int c=0; c<num_training; c++) //对训练集的遍历 
			{
				double dis=0 ;			// 新集第r篇与训练集第c篇 
				double len1 = 0, len2 = 0, dot_product = 0 ; // 辅助计算余弦距离 
				for( int j=0; j<word.size(); j++) 
				{	
					// 欧式距离 
					dis += fabs(OneHot_training[c][j]-OneHot_new[r][j]);									
				}
				dis = sqrt( dis ) ;				
				knn.insert( multimap<double,int>::value_type(dis,c)) ;
				// 插入测试集第r篇与训练集第c篇 （距离，标签），自动按照距离排序 
			}
			
			for( int i=0; i<6; i++)
				pro_emo[i] = 0 ;   // 每篇文章6种情感概率初始化为1 
			
			string label[6]={ "anger", "disgust", "fear", "joy", "sad", "surprise" } ;
			multimap<double,int>::iterator knnit = knn.begin();	
			multimap<double,int>::iterator knnend = knn.end();	
			if( knnit->first == 0)
			{
				int c = knnit->second ;
				ofile_new <<r+1 <<"," <<training[c].pro[0] <<"," <<training[c].pro[1] <<"," <<training[c].pro[2] 
					<<"," <<training[c].pro[3] <<"," <<training[c].pro[4] <<"," <<training[c].pro[5]<<endl ;	
				continue ;			 
			} 
			else
			{
				knnit = knn.begin();	
				double pro_sum = 0;
				for( int i=0; i<6; i++)
				{
					knnit = knn.begin();	
							
					for( int x=0; x<k,knnit!=knnend; x++, knnit++)
					{
						pro_emo[i] += training[ knnit->second ].pro[i] / knnit->first  ;
					}
				}
				//概率归一化 
				for( int i=0; i<6; i++)
				{
					pro_sum += pro_emo[i]; 
				}			
				for( int i=0; i<6; i++)
				{
					pro_emo[i] /= pro_sum ;	
				}	
				ofile_new <<r+1 <<"," <<pro_emo[0] <<"," <<pro_emo[1] <<"," <<pro_emo[2] 
							<<"," <<pro_emo[3] <<"," <<pro_emo[4] <<"," <<pro_emo[5]<<endl ;
			}
				
		} 
	}
} 




	
