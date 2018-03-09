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

double alpha = 0.6 ;
vector<string> word;    //存training单词 
vector<Passage> training ; //训练集的所有文章 
double emotion[6] ;  // 每种情感的概率 = 标签为Ei的文章数量/文章总数 
int Times_training[PASSAGE_NUM][WORD_NUM] ;  	// 训练集每篇文章i的单词j出现次数表 
double Probability[6][WORD_NUM] ; //6种情感中每个单词分别出现的总次数
int all_sum[6] ;			// 每一种情感包含的单词总数
// PASSAGE_NUM 3000训练集文章上限 ,WORD_NUM 60000单词上限 

int main()
{
	/** 打开training文件，读取文件，分割为文本和标签 **/ 	 
	ifstream file0;   
	//file0.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nbSMALL\\train_set.csv");
	file0.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\分类原始数据\\train_set.csv");

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
	
	int num_training = ( SplitVec0.size() - 1 ) / 2 ;	//训练集文章数量 	
	/** FINISHED 打开training文件，读取文件，分割为文本和标签 **/
	
	/* 构造训练集单词次数矩阵，即计算每篇文章各个单词出现次数 */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//构造第i篇文章  	 
		p.label = SplitVec0[2*i+1] ; 	//每一篇文章的label 
		string tmp = SplitVec0[2*i] ;       	//第i篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第i篇文章的单词
		training.push_back( p ) ;		//第i篇文章插入training文章vector 

		//构造第i行的Times矩阵 
		for( int j=0; j<p.voc.size(); j++ ) // 查看文章中每个单词 
		{
			string subtmp = p.voc[j] ;	   // 第i篇文章的第j个单词 
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //单词表已经存在这个单词 
			{
				int loc = iterloc-word.begin() ;  	// 第j个单词所在单词库位置 
				Times_training[i][loc] ++;      // 次数+1 
			}
			else //新单词 
			{	
				word.push_back( subtmp ) ;			// 插入新单词 
				Times_training[i][ word.size()-1 ] = 1; 	// 第j个单词所在单词库位置 ，// 次数=1 
			}		
		}				
	}	
	/* FINISHED 构造训练集单词次数矩阵 */ 
		
	/* 计算第i种情感下，出现单词j的概率 矩阵prob=P(Xj,Ei)
	每个单词分别在6种情感的总次数，每一种情感出现的单词总数，每一种情感出现的不重复单词数目*/ 
	
	/* 构造一个矩阵Probability[i][j], 第i种情感中单词j出现次数 */ 
	string label[6]={ "anger", "disgust", "fear", "joy", "sad", "surprise" } ;	
	// 分类的取值， 6种情感
	for( int i=0; i<6; i++)		//第i种情感标签 
	{	// 遍历训练集，找出标签为Ei的文章 
		for( int t=0; t<num_training; t++)  //训练集中的第t篇文章 
		{	// vector<Passage> training 是训练集的所有文章 
			if( training[t].label == label[i] )  
			{	// 如果第t篇文章 属于 第i种情感
				emotion[i] ++ ;  // 情感为Ei的文章数量+1 				
				for( int j=0; j<word.size(); j++)  //遍历所有单词 
				{	// Times_training[t][j] 是训练集第t篇文章的单词j出现次数  					
					Probability[i][j] += Times_training[t][j] ;
				}	
			}
		}						
	}  /* FINISHED  构造矩阵Probability[i][j] */  
	
	/* 构造一个向量all_sum[i], 第i种情感中单词出现总次数（重复的也算） */  
	for( int i=0; i<6; i++)
	{
		for( int j=0; j<word.size(); j++)
		{	// 累加单词j在第i种情感中出现次数		
			all_sum[i] += Probability[i][j] ; 
		}
	} /* FINISHED 构造一个向量all_sum[i] */ 
	
	/*为了计算方便，把 Probability[i][j]第i种情感中单词j出现总次数
	【改成】 P(Xj|Ei) */
	for( int i=0; i<6; i++)
	{
		for( int j=0; j<word.size(); j++)
		{   // 拉普拉斯平滑公式
			Probability[i][j] = (Probability[i][j]+alpha) / ( all_sum[i] + alpha*word.size()) ;	
		}
	}

	/** 打开validation文件，读取文件，分割为文本和标签 **/ 	   
	ifstream file1; 
	//file1.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\nbSMALL\\test_set.csv");
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
	int num_new = ( SplitVec1.size() - 1 ) / 2 ;	//validation文章数量 	
	/** FINISHED 打开validation文件，读取文件，分割为文本和标签 **/
	
	
	/* * * * * * * * * * * * * * * 大数据 * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //定义输出文件OneHot.txt
	//ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\classification_dataset\\small\\15352204_Sample_KNN_classification.csv");    //作为输出文件打开	
	ofile_new.open("D:\\i学习\\大三\\人工智能 饶洋辉\\Lab2\\DATA\\分类原始数据\\15352204_LinDan_NB_classification.csv");   
	
	ofile_new <<"textid" <<"," <<"label" <<endl ;
	double num_correct = 0;
		
	/* 分割单词时，NB分类计算概率 */ 
	double pro[6] ;  // 每种情感的概率	
	for (int x = 0; x<num_new; x++)	//测试集的第x篇文章 
	{
		Passage p;      				//构造第x篇文章  	 
		p.label = SplitVec1[2*x+1] ; 	//每一篇文章的label 
		string tmp = SplitVec1[2*x] ;   //第x篇文章没有截断前 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //构造第x篇文章的单词
		// 每篇文章的情感概率初始化 P(Ei) 
		for( int i=0; i<6; i++)
			pro[i] = 1.0 * emotion[i] / num_training;  
			
		double max_pro = 0 ; 	// 情感概率最大值 
		int max_i = 0;			// 情感概率最大值对应情感下标 
		for( int j=0; j<p.voc.size(); j++ ) // 查看文章中每个单词
		{ 	
			string subtmp = p.voc[j] ;  // 第x篇文章的第j个单词 	   
			//单词是否已经存在单词库 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 			
			if( iterloc!= word.end() )  //单词表已经存在这个单词 
			{
				int loc = iterloc-word.begin() ;  	//第j个单词所在单词库位置 
				for( int i=0; i<6; i++)
					pro[i] *= Probability[i][loc] ;
				// 根据NB分类公式，累积 P(Xj|Ei)P(Ei)
			}
			else //新单词,原本概率为0，进行拉普拉斯平滑 
			{	
				for( int i=0; i<6; i++)
				{	//拉普拉斯平滑 
					pro[i] = pro[i] * alpha / ( all_sum[i] + word.size()) ;
				}					
			}		
		}
				
		// 计算测试集第x篇文章，哪一种情感概率高 
		for( int i=0; i<6; i++)
		{
			if( pro[i]>max_pro && pro[i]<=1 )
			{
				max_pro = pro[i] ;
				max_i = i ;
			}
		}
		ofile_new << x+1 <<"," <<label[max_i] <<endl ;
		if( label[max_i] == p.label )	
		{	// 预测结果==正确答案？ 
			num_correct ++ ; //计算准确率 
		}	
	}
	/* FINISHED */ 
	
	ofile_new.close() ;
	double rate = 1.0 * num_correct / num_new ;
	//cout <<"alpha: "<<alpha <<" num_correct: " <<num_correct <<" rate: " <<rate <<endl ;
}
	
	
	


