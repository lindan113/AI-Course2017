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
#define WORD_SIZE 50000 // 文章字符上限 
#define PASSAGE_NUM 1000	// 训练集文章上限 
#define WORD_NUM 60000	// 单词上限 
using namespace std;
using namespace boost;

class Passage
{
public: 
	// anger,disgust,fear,joy,sad,surprise
	double pro[6] ;   		//每一篇文章的情感概率 
	vector<string> voc ;	//每一篇文章的切割结果（有重复） 
	
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

double alpha = 0.04 ;
vector<string> word;    //存training单词 
vector<Passage> training ; //训练集的所有文章 
map<Pos, double> TF_map; // 训练集词频（稀疏矩阵） 
double TF_training[PASSAGE_NUM][WORD_NUM] ; // 训练集词频(二维数组) 
double pro_tmp[PASSAGE_NUM][6] ; 

int main()
{
	/** 打开training文件，读取文件，分割为文本和标签 **/ 	 
	ifstream file0;   
	file0.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\train_set.csv");
	//file0.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nb_reg_small\\train_set.csv");

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
	
	/* 计算单词在训练集各个出现次数 TF_map辅助构造TF矩阵 */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      
		for( int u=0; u<=5; u++)
		{	// string类型转double类型		
			p.pro[u]= std::stod( SplitVec0[7*i+1+u] ) ;	 //第i篇文章的情感概率		
		}
				
		string tmp = SplitVec0[7*i] ;       	//第i篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第i篇文章的单词
		training.push_back( p ) ;		//第i篇文章插入training文章vector 

		//构造第i行的TF_map矩阵 
		for( int j=0; j<p.voc.size(); j++ ) // 查看文章中每个单词 
		{
			string subtmp = p.voc[j] ;	   // 第i篇文章的第j个单词 
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //单词表已经存在这个单词 
			{
				int loc = iterloc-word.begin() ;  	// 第j个单词所在单词库位置 
				int oldVal = TF_map[ Pos(i, loc )] ; //矩阵位置的员原来的值  
				TF_map[ Pos(i, loc )] = oldVal+1 ;  
				//单词再一次出现，修改map中这个矩阵位置的值
			}
			else //新单词 
			{	
				word.push_back( subtmp ) ;		// 插入新单词 
				TF_map[ Pos(i, word.size()-1 ) ] = 1 ;	
			}		
		}				
	}
	/* FINISHED  计算单词在训练集各个出现次数 */ 
	
	/* 稀疏矩阵三元组转成二维数组  */
	map<Pos, double>::iterator it = TF_map.begin();  
	map<Pos, double>::iterator end = TF_map.end();
	for( ; it!=end; it++ ) 
		TF_training[it->first.row][it->first.col] = it->second ;			
	/* FINISHED 稀疏矩阵三元组转成二维数组  */

	/* 单词在训练集的次数矩阵->TF词频矩阵 laplace平滑 */ 
	for( int i=0; i<num_training; i++ )
	{	// 每篇文章单词共同的分母 
		double denominator = training[it->first.row].voc.size() + alpha*word.size() ;
		for( int j=0; j<word.size(); j++ )
			TF_training[i][j] = ( TF_training[i][j] + alpha )/denominator;
	}	
	/* FINISHED  单词在训练集的次数矩阵->TF词频矩阵  */
		
	/** 打开validation文件，读取文件，分割为文本和标签 **/ 	   
	ifstream file1; 
	file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\test_set.csv");
	//file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\validation_set.csv");
	//file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nb_reg_small\\test_set.csv");	
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //从文件中读取字符到字符串str1，当遇到字符'\0'或读取了SIZE个字符时终止。
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //分割字符串 以Tab和换行为分隔符
	vector<string>::iterator begin1 = SplitVec1.begin() ;
	for( int i=0; i<7; i++ )
		SplitVec1.erase(begin1) ;		// 删除第一行的七项 

	int num_new = SplitVec1.size() / 7 ;	//validation文章数量 	
	/** FINISHED 打开validation文件，读取文件，分割为文本和标签 **/
	
	/* * * * * * * * * * * * * * * 大数据 * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //定义输出文件OneHot.txt
	ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\15352204_LinDan_NB_regression.csv");    //作为输出文件打开	
	//ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\回归原始数据\\15352204_LinDan_NB_regression.csv");
	//ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nb_reg_small\\validation相关度评估.xlsx");   
	ofile_new <<"id" <<"," <<"anger" <<"," <<"disgust" <<"," <<"fear"
					<<"," <<"joy" <<"," <<"sad" <<"," <<"surprise" <<endl ;
					
	
	for (int x = 0; x<num_new; x++)	//测试集的第x篇文章 
	{
		Passage p;      //构造第x篇文章  	 
		string tmp = SplitVec1[7*x] ;       	//第x篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第x篇文章的单词
		
		/* 构造pro_tmp[t][i] 第t篇文章第i种情感的累加分量  ∏[P(Xj|dt )P(Ei)]*/ 
		for( int t=0; t<num_training; t++)
		{
			for( int i=0; i<6; i++ )
			{	// 初始化为P(Ei)
				pro_tmp[t][i] = training[t].pro[i] ;  
			}  
		}				
		double pro_sum = 0 ;  // 每篇文章6种情感概率求和,初始化为0 (为归一化）		
		for( int j=0; j<p.voc.size(); j++ ) //文章句子里的第j个单词 	
		{	// 遍历训练集，找出标签为Ei的文章 						
			string subtmp = p.voc[j] ;	   // 第x篇文章的第j个单词					
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find(word.begin(),word.end(),subtmp); 
			vector<string>::iterator wordend = word.end() ;			
			for( int t=0; t<num_training; t++)  //训练集中的第t篇文章			
			{													
				for( int i=0; i<6; i++)		//第x篇文章的第i种情感标签  
				{													
					if( iterloc!= wordend )  //单词表已经存在这个单词 
					{
						int loc = iterloc-word.begin(); //第j个单词所在单词库位置 
						pro_tmp[t][i] *= TF_training[t][loc] ;						
					}
					else //新单词 
					{	
						pro_tmp[t][i] = pro_tmp[t][i] * alpha / 
							( training[t].voc.size() + alpha*word.size()) ;
						// 拉普拉斯平滑	
					}										
				}		
			}							
		} /* FINISHED 构造pro_tmp[t][i] */
		
		/* 情感为Ei的概率=对所有文章的累积分量求和 */
		for( int t=0; t<num_training; t++)
		{
			for( int i=0; i<6; i++)
			{	//得到该文章为情感为Ei的概率				
				p.pro[i] += pro_tmp[t][i] ;
			}
		} 
		/* FINISHED 该文章情感为Ei的概率 */
		/*  6种情感概率归一化 */
		for( int i=0; i<6; i++)
		{	
			pro_sum += p.pro[i] ;
		}	
		for( int i=0; i<6; i++ )
		{
			p.pro[i] /= pro_sum ;
		} 
		/* FINISHED 6种情感概率归一化 */
		ofile_new <<x+1 <<"," <<p.pro[0] <<"," <<p.pro[1] <<"," <<p.pro[2] 
						<<"," <<p.pro[3] <<"," <<p.pro[4] <<"," <<p.pro[5] <<endl ;						
	} // end for(int x = 0; x==0; x++)	//新集的第x篇文章 		
}
	
	
	
	


