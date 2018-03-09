/*
ע���޸��ļ�·��
test_set.csv��Ҫ�ֶ�ɾ����һ�� 
*/ 
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>
#define SIZE 1000000 // �����ַ�����  
#define WORD_SIZE 500000 // �����ַ����� 
#define PASSAGE_NUM 1000	// ѵ������������ 
#define WORD_NUM 60000	// �������� 

using namespace std;
using namespace boost;

class Passage
{
public: 
	// anger,disgust,fear,joy,sad,surprise
	double pro[6] ;   		//ÿһƪ���µ�label 
	vector<string> voc ;	//ÿһƪ���µĵ��ʿ� 
	
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
	//����˳������ 
	bool operator<( const Pos& p2) const
	{
		if( this->row != p2.row )
			return this->row < p2.row ;
		else // this->row == t2.row 
		{
			if( this->col==p2.col ) return false ;
			else return this->col<p2.col ; //��С��������
		}
	}
};

vector<string> word;    //��training���� 
vector<Passage> training ; //ѵ�������������� 
vector<Passage> newdata ;  //���Լ����������� 
map<Pos, double> TF0, TF1;  //��������TF�����map

int OneHot_training[PASSAGE_NUM][WORD_NUM] ; //ѵ����OneHot_training���� 
double TF_training[PASSAGE_NUM][WORD_NUM] ; //ѵ����TF_training���� 
int OneHot_new[PASSAGE_NUM][WORD_NUM] ; //�µ����ݼ�(��֤��/���Լ�)OneHot ���� 
double TF_new[PASSAGE_NUM][WORD_NUM] ; //�µ����ݼ�(��֤��/���Լ�)OneHot ���� 
// PASSAGE_NUM 3000ѵ������������ ,WORD_NUM 60000�������� 

int main()
{
	/** ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	 
	ifstream file0;   
	file0.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\train_set.csv");
	char *str0;	
	str0 = new char[SIZE] ;	
	file0.get(str0,SIZE,'\0');     //���ļ��ж�ȡ�ַ����ַ���str1���������ַ�'\0'���ȡ��SIZE���ַ�ʱ��ֹ��
	string strTmp0 = str0;
	file0.close();	
	vector<string> SplitVec0;
	split(SplitVec0, strTmp0, is_any_of(",\n"), token_compress_on ); //�ָ�ѵ�����ı��ַ���
	vector<string>::iterator begin = SplitVec0.begin() ;
	for( int i=0; i<7; i++ ) //��7��Ҫɾ�� 
		SplitVec0.erase(begin) ;		// ɾ����һ�е����� 
	
	int num_training = SplitVec0.size() / 7 ;	//�������� 	
	/** FINISHED ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/

	
	/* ����ѵ����OneHot�����TF���� */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//�����iƪ����  	 
		for( int u=0; u<=5; u++)
		{	// string����תdouble����		
			p.pro[u]= std::stod( SplitVec0[7*i+1+u] ) ;	 //��iƪ���µ���и���		
		}	
		string tmp = SplitVec0[7*i] ;       	//��iƪ����û�нض�ǰ 	
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����iƪ���µĵ���
		training.push_back( p ) ;		//��iƪ���²���training����vector 

		//�����i�е�OneHot��TF���� 
		for( int j=0; j<p.voc.size(); j++ ) // �鿴������ÿ������ 
		{
			string subtmp = p.voc[j] ;	   // ��iƪ���µĵ�j������ 
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //���ʱ��Ѿ������������ 
			{
				int loc = iterloc-word.begin() ;  	// ��j���������ڵ��ʿ�λ�� 
				OneHot_training[i][loc] = 1;
				int oldVal = TF0[ Pos(i, loc )] ; //����λ�õ�Աԭ����ֵ  
				TF0[ Pos(i, loc )] = oldVal+1 ;  //������һ�γ��֣��޸�map���������λ�õ�ֵ
			}
			else //�µ��� 
			{	
				word.push_back( subtmp ) ;			// �����µ��� 
				OneHot_training[i][ word.size()-1 ] = 1; 	// ��j���������ڵ��ʿ�λ�� 
				TF0[ Pos(i, word.size()-1 ) ] = 1 ;	  //�´ʵ�һ�γ���
			}		
		}				
	}
	
	/* FINISHED ����training����vector, OneHot_training���� */ 
	
	/* ϡ�����TFת�ɶ�ά���� */
	//���ִ���->TF 
	map<Pos, double>::iterator it = TF0.begin();  
	map<Pos, double>::iterator end = TF0.end();
	for( ; it!=end; it++)
	{	// TF= ������ֵĴ���/�����ܵ�����  
		it->second = it->second / training[it->first.row].voc.size() ;
	}	
	it = TF0.begin();
	for( ; it!=end; it++ ) 
	{
		TF_training[it->first.row][it->first.col] = it->second ;		
	}
	/* FINISHED ϡ�����TFת�ɶ�ά���� */
	
	/** ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	   
	ifstream file1; 
	file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\test_set.csv");
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //���ļ��ж�ȡ�ַ����ַ���str1���������ַ�'\0'���ȡ��SIZE���ַ�ʱ��ֹ��
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //�ָ��ַ��� ��Tab�ͻ���Ϊ�ָ���
	begin = SplitVec1.begin() ;
	for( int i=0; i<7; i++ )
		SplitVec1.erase(begin) ;		// ɾ����һ�е����� 

	int num_new = SplitVec1.size() / 7 ;	//validation�������� 	
	/** FINISHED ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/

	/* ����validation����OneHot_training���� */ 
	
	for (int i = 0; i<num_new; i++)	//��iƪ���� 
	{
		Passage p;      				//�����iƪ����  	 
		string tmp = SplitVec1[7*i] ;       	//��iƪ����û�нض�ǰ 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����iƪ���µĵ���
		newdata.push_back( p ) ;		//��iƪ���²���training����vector 
		
		for( int j=0; j<p.voc.size(); j++ ) //��j������ 
		{ // �鿴������ÿ������ 
			string subtmp = p.voc[j] ;	   // ��iƪ���µĵ�j������ 
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //���ʱ��Ѿ������������ 
			{
				int loc = iterloc-word.begin() ;  	// ��j���������ڵ��ʿ�λ�� 
				//OneHot0[ Pos(i, loc ) ] = 1 ;  //����Ϊ1 	
				OneHot_new[i][loc] = 1;
				int oldVal = TF1[ Pos(i, loc )] ; //����λ�õ�Աԭ����ֵ  
				TF1[ Pos(i, loc )] = oldVal+1 ;  //������һ�γ��֣��޸�map���������λ�õ�ֵ
			}
			else //�µ��� 
			{	
				word.push_back( subtmp ) ;			// �����µ��� 
				OneHot_new[i][ word.size()-1 ] = 1; 	// ��j���������ڵ��ʿ�λ�� 
				//OneHot0[ Pos(i, word.size()-1 ) ] = 1 ;  // OneHot ���� 
				TF1[ Pos(i, word.size()-1 ) ] = 1 ;	   // Ƶ�ξ��� 
			}		
		}
						
	}
	/* FINISHED ����training����vector, OneHot_training���� */ 
	
	/* ϡ�����TFת�ɶ�ά���� */
	//���ִ���->TF 
	map<Pos, double>::iterator it1 = TF1.begin();  
	map<Pos, double>::iterator end1 = TF1.end();
	int times ;
	for( ; it1!=end1; it1++)
	{	// TF= ������ֵĴ���/�����ܵ�����  
		times = newdata[it1->first.row].voc.size() ;
		it1->second = it1->second / times;
	}		
	it1 = TF1.begin();
	for( ; it1!=end1; it1++ ) 
	{
		TF_new[it1->first.row][it1->first.col] = it1->second ;		
	}
	/* ϡ�����TFת�ɶ�ά���� */
	
	
	/* KNN ���� */ 
	/* * * * * * * * * * * * * * * ������ * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //��������ļ�OneHot.txt
	ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\15352204_LinDan_KNN_regression.csv");    //��Ϊ����ļ���	
	//ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nb_reg_small\\validation��ض�����.xlsx");   
	ofile_new <<"id" <<"," <<"anger" <<"," <<"disgust" <<"," <<"fear"
					<<"," <<"joy" <<"," <<"sad" <<"," <<"surprise" <<endl ;	
	
	double pro_emo[6] ; // ÿƪ����6����и��� 
	for( int k=15; k==15; k++ )    //KNN�㷨��ͬKֵ 
	{
		for( int r=0; r<num_new; r++)  //�µ����ݼ�ÿһƪ���£���rƪ���� 
		{						
			//multimap<double,int,greater<double>> knn ; //���Ҿ��� 
			multimap<double,int,less<double>> knn ;     //ŷʽ����
			//�Բ��Լ�ÿһƪ������һ��knn������ǰk��������������£����룬��ǩ��
			for( int c=0; c<num_training; c++) //��ѵ�����ı��� 
			{
				double dis=0 ;			// �¼���rƪ��ѵ������cƪ 
				double len1 = 0, len2 = 0, dot_product = 0 ; // �����������Ҿ��� 
				for( int j=0; j<word.size(); j++) 
				{	
					// ŷʽ���� 
					dis += fabs(OneHot_training[c][j]-OneHot_new[r][j]);									
				}
				dis = sqrt( dis ) ;				
				knn.insert( multimap<double,int>::value_type(dis,c)) ;
				// ������Լ���rƪ��ѵ������cƪ �����룬��ǩ�����Զ����վ������� 
			}
			
			for( int i=0; i<6; i++)
				pro_emo[i] = 0 ;   // ÿƪ����6����и��ʳ�ʼ��Ϊ1 
			
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
				//���ʹ�һ�� 
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




	
