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
#define WORD_SIZE 50000 // �����ַ����� 
#define PASSAGE_NUM 1000	// ѵ������������ 
#define WORD_NUM 60000	// �������� 
using namespace std;
using namespace boost;

class Passage
{
public: 
	// anger,disgust,fear,joy,sad,surprise
	double pro[6] ;   		//ÿһƪ���µ���и��� 
	vector<string> voc ;	//ÿһƪ���µ��и��������ظ��� 
	
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

double alpha = 0.04 ;
vector<string> word;    //��training���� 
vector<Passage> training ; //ѵ�������������� 
map<Pos, double> TF_map; // ѵ������Ƶ��ϡ����� 
double TF_training[PASSAGE_NUM][WORD_NUM] ; // ѵ������Ƶ(��ά����) 
double pro_tmp[PASSAGE_NUM][6] ; 

int main()
{
	/** ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	 
	ifstream file0;   
	file0.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\train_set.csv");
	//file0.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nb_reg_small\\train_set.csv");

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
	
	/* ���㵥����ѵ�����������ִ��� TF_map��������TF���� */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      
		for( int u=0; u<=5; u++)
		{	// string����תdouble����		
			p.pro[u]= std::stod( SplitVec0[7*i+1+u] ) ;	 //��iƪ���µ���и���		
		}
				
		string tmp = SplitVec0[7*i] ;       	//��iƪ����û�нض�ǰ 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����iƪ���µĵ���
		training.push_back( p ) ;		//��iƪ���²���training����vector 

		//�����i�е�TF_map���� 
		for( int j=0; j<p.voc.size(); j++ ) // �鿴������ÿ������ 
		{
			string subtmp = p.voc[j] ;	   // ��iƪ���µĵ�j������ 
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //���ʱ��Ѿ������������ 
			{
				int loc = iterloc-word.begin() ;  	// ��j���������ڵ��ʿ�λ�� 
				int oldVal = TF_map[ Pos(i, loc )] ; //����λ�õ�Աԭ����ֵ  
				TF_map[ Pos(i, loc )] = oldVal+1 ;  
				//������һ�γ��֣��޸�map���������λ�õ�ֵ
			}
			else //�µ��� 
			{	
				word.push_back( subtmp ) ;		// �����µ��� 
				TF_map[ Pos(i, word.size()-1 ) ] = 1 ;	
			}		
		}				
	}
	/* FINISHED  ���㵥����ѵ�����������ִ��� */ 
	
	/* ϡ�������Ԫ��ת�ɶ�ά����  */
	map<Pos, double>::iterator it = TF_map.begin();  
	map<Pos, double>::iterator end = TF_map.end();
	for( ; it!=end; it++ ) 
		TF_training[it->first.row][it->first.col] = it->second ;			
	/* FINISHED ϡ�������Ԫ��ת�ɶ�ά����  */

	/* ������ѵ�����Ĵ�������->TF��Ƶ���� laplaceƽ�� */ 
	for( int i=0; i<num_training; i++ )
	{	// ÿƪ���µ��ʹ�ͬ�ķ�ĸ 
		double denominator = training[it->first.row].voc.size() + alpha*word.size() ;
		for( int j=0; j<word.size(); j++ )
			TF_training[i][j] = ( TF_training[i][j] + alpha )/denominator;
	}	
	/* FINISHED  ������ѵ�����Ĵ�������->TF��Ƶ����  */
		
	/** ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	   
	ifstream file1; 
	file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\test_set.csv");
	//file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\validation_set.csv");
	//file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nb_reg_small\\test_set.csv");	
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //���ļ��ж�ȡ�ַ����ַ���str1���������ַ�'\0'���ȡ��SIZE���ַ�ʱ��ֹ��
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //�ָ��ַ��� ��Tab�ͻ���Ϊ�ָ���
	vector<string>::iterator begin1 = SplitVec1.begin() ;
	for( int i=0; i<7; i++ )
		SplitVec1.erase(begin1) ;		// ɾ����һ�е����� 

	int num_new = SplitVec1.size() / 7 ;	//validation�������� 	
	/** FINISHED ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/
	
	/* * * * * * * * * * * * * * * ������ * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //��������ļ�OneHot.txt
	ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\15352204_LinDan_NB_regression.csv");    //��Ϊ����ļ���	
	//ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\�ع�ԭʼ����\\15352204_LinDan_NB_regression.csv");
	//ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nb_reg_small\\validation��ض�����.xlsx");   
	ofile_new <<"id" <<"," <<"anger" <<"," <<"disgust" <<"," <<"fear"
					<<"," <<"joy" <<"," <<"sad" <<"," <<"surprise" <<endl ;
					
	
	for (int x = 0; x<num_new; x++)	//���Լ��ĵ�xƪ���� 
	{
		Passage p;      //�����xƪ����  	 
		string tmp = SplitVec1[7*x] ;       	//��xƪ����û�нض�ǰ 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����xƪ���µĵ���
		
		/* ����pro_tmp[t][i] ��tƪ���µ�i����е��ۼӷ���  ��[P(Xj|dt )P(Ei)]*/ 
		for( int t=0; t<num_training; t++)
		{
			for( int i=0; i<6; i++ )
			{	// ��ʼ��ΪP(Ei)
				pro_tmp[t][i] = training[t].pro[i] ;  
			}  
		}				
		double pro_sum = 0 ;  // ÿƪ����6����и������,��ʼ��Ϊ0 (Ϊ��һ����		
		for( int j=0; j<p.voc.size(); j++ ) //���¾�����ĵ�j������ 	
		{	// ����ѵ�������ҳ���ǩΪEi������ 						
			string subtmp = p.voc[j] ;	   // ��xƪ���µĵ�j������					
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find(word.begin(),word.end(),subtmp); 
			vector<string>::iterator wordend = word.end() ;			
			for( int t=0; t<num_training; t++)  //ѵ�����еĵ�tƪ����			
			{													
				for( int i=0; i<6; i++)		//��xƪ���µĵ�i����б�ǩ  
				{													
					if( iterloc!= wordend )  //���ʱ��Ѿ������������ 
					{
						int loc = iterloc-word.begin(); //��j���������ڵ��ʿ�λ�� 
						pro_tmp[t][i] *= TF_training[t][loc] ;						
					}
					else //�µ��� 
					{	
						pro_tmp[t][i] = pro_tmp[t][i] * alpha / 
							( training[t].voc.size() + alpha*word.size()) ;
						// ������˹ƽ��	
					}										
				}		
			}							
		} /* FINISHED ����pro_tmp[t][i] */
		
		/* ���ΪEi�ĸ���=���������µ��ۻ�������� */
		for( int t=0; t<num_training; t++)
		{
			for( int i=0; i<6; i++)
			{	//�õ�������Ϊ���ΪEi�ĸ���				
				p.pro[i] += pro_tmp[t][i] ;
			}
		} 
		/* FINISHED ���������ΪEi�ĸ��� */
		/*  6����и��ʹ�һ�� */
		for( int i=0; i<6; i++)
		{	
			pro_sum += p.pro[i] ;
		}	
		for( int i=0; i<6; i++ )
		{
			p.pro[i] /= pro_sum ;
		} 
		/* FINISHED 6����и��ʹ�һ�� */
		ofile_new <<x+1 <<"," <<p.pro[0] <<"," <<p.pro[1] <<"," <<p.pro[2] 
						<<"," <<p.pro[3] <<"," <<p.pro[4] <<"," <<p.pro[5] <<endl ;						
	} // end for(int x = 0; x==0; x++)	//�¼��ĵ�xƪ���� 		
}
	
	
	
	


