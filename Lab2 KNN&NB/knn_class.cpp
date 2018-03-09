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
	string label ;   		//ÿһƪ���µ�label 
	vector<string> voc ;	//ÿһƪ���µĵ��ʿ� 
	
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
	file0.open("D:\\�ĵ�\\MATLAB\\BPNN_train_small.csv");
	char *str0;	
	str0 = new char[SIZE] ;	
	file0.get(str0,SIZE,'\0');     //���ļ��ж�ȡ�ַ����ַ���str1���������ַ�'\0'���ȡ��SIZE���ַ�ʱ��ֹ��
	string strTmp0 = str0;
	file0.close();	
	vector<string> SplitVec0;
	split(SplitVec0, strTmp0, is_any_of(",\n"), token_compress_on ); //�ָ�ѵ�����ı��ַ���
	vector<string>::iterator begin = SplitVec0.begin() ;
	SplitVec0.erase(begin) ;
	SplitVec0.erase(begin) ;
	
	int num_training = SplitVec0.size() / 2 ;	//�������� 	
	/** FINISHED ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/
	
	/* ����ѵ����OneHot�����TF���� */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//�����iƪ����  	 
		p.label = SplitVec0[2*i+1] ; 	//ÿһƪ���µ�label 
		string tmp = SplitVec0[2*i] ;       	//��iƪ����û�нض�ǰ 		
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
	file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\����ԭʼ����\\test_set.csv");
	char* str1 = new char[SIZE] ;	
	file1.get(str1,SIZE,'\0');     //���ļ��ж�ȡ�ַ����ַ���str1���������ַ�'\0'���ȡ��SIZE���ַ�ʱ��ֹ��
	string strTmp1 = str1;
	file1.close();	
	vector<string> SplitVec1;
	split(SplitVec1, strTmp1, is_any_of(",\n"), token_compress_on ); //�ָ��ַ��� ��Tab�ͻ���Ϊ�ָ���
	begin = SplitVec1.begin() ;
	SplitVec1.erase(begin) ;
	SplitVec1.erase(begin) ;
	int num_new = SplitVec1.size() / 2 ;	//validation�������� 	
	/** FINISHED ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/

	/* ����validation����OneHot_training���� */ 
	
	for (int i = 0; i<num_new; i++)	//��iƪ���� 
	{
		Passage p;      				//�����iƪ����  	 
		p.label = SplitVec1[2*i+1] ; 	//ÿһƪ���µ�label 
		string tmp = SplitVec1[2*i] ;       	//��iƪ����û�нض�ǰ 		
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
				OneHot_new[i][loc] = 1;
				int oldVal = TF1[ Pos(i, loc )] ; //����λ�õ�Աԭ����ֵ  
				TF1[ Pos(i, loc )] = oldVal+1 ;  //������һ�γ��֣��޸�map���������λ�õ�ֵ
			}
			else //�µ��� 
			{	
				word.push_back( subtmp ) ;			// �����µ��� 
				OneHot_new[i][ word.size()-1 ] = 1; 	// ��j���������ڵ��ʿ�λ�� 
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
	ofstream ofile_new;               //��������ļ�OneHot.txt
	ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\����ԭʼ����\\15352204_LinDan_KNN_classification.csv");    //��Ϊ����ļ���	
	ofile_new <<"textid" <<"," <<"label" <<endl ;
	
	double max_rate = 0;    // �Բ�ͬ��K,����׼ȷ�����ֵ 
	int max_rate_k = 0;     // ׼ȷ�����ֵ��Ӧ��K 
	for( int k=14; k<=14; k++ )    //KNN�㷨��ͬKֵ 
	{
		int num_correct = 0;		//���Խ��׼ȷ�ĸ��� 
		for( int r=0; r<num_new; r++)  //�µ����ݼ�ÿһƪ���£���rƪ���� 
		{
			string answer = SplitVec1[2*r+1] ; 	//�¼���rƪ���µ�label ����ȷ�� 
			multimap<double,string,greater<double>> knn ;  
			//�Բ��Լ�ÿһƪ������һ��knn������ǰk��������������£����룬��ǩ���� 
			
			for( int c=0; c<num_training; c++) //��ѵ�����ı��� 
			{
				double dis=0 ;			// �¼���rƪ��ѵ������cƪ 
				double len1 = 0, len2 = 0, dot_product = 0 ;
				for( int j=0; j<word.size(); j++) 
				{	// ŷʽ���� 
					// dis += ( TF_training[c][j] - TF_new[r][j] )*( TF_training[c][j] - TF_new[r][j] ) ;	
					len1 += OneHot_training[c][j] ;
					len2 += OneHot_new[r][j] ;
					dot_product += OneHot_training[c][j] * OneHot_new[r][j] ;								
				}
				dis= dot_product / sqrt( len1 * len2 ) ;
				knn.insert( multimap<double,string>::value_type(dis,SplitVec0[2*c+1])) ;
				// ������Լ���rƪ��ѵ������cƪ �����룬��ǩ�����Զ����վ������� 
			}					
			
			int count[6] ;
			for(int i=0; i<6; i++)
				count[i] = 0;   
			// �Ծ��������K��ѵ�������½��С�����ͶƱ�� 			
			string label[6]={ "anger", "disgust", "fear", "joy", "sad", "surprise" } ;
			multimap<double,string>::iterator knnit = knn.begin();			
			for( int i=0; i<k; i++, knnit++)
			{
				string tmp = knnit->second ;
				// ��ǰK��ѵ�������µı�ǩ���� 
				if( tmp=="anger") 			count[0]++ ;
				else if( tmp=="disgust") 	count[1]++ ;
				else if( tmp=="fear") 		count[2]++ ;
				else if( tmp=="joy") 		count[3]++ ;
				else if( tmp=="sad") 		count[4]++ ;
				else if( tmp=="surprise") 	count[5]++ ;
				else cout<<endl<<"error:" <<knnit->second <<endl ;
			}
			int maxNum = 0;		//ǰk�����ֵı�ǩ���������ֵ   
			int max_i = 0; 		//���ֵ��Ӧ�ı�ǩ�±� 
			//�ҳ�ǰk�����ֵı�ǩ���������ֵ  
			for( int i=0; i<6; i++ )
			{
				if ( count[i]>maxNum )
				{
					maxNum = count[i] ;
					max_i = i;
				} 
			}
			string predict = label[max_i] ; 	
			//�������ֵ��Ӧ�ı�ǩ�±꣬�ҵ�Ԥ��ı�ǩ��� 		
			ofile_new <<r+1 <<"," <<predict <<endl ; 
			if( predict == answer ) // ��֤ 
				num_correct ++ ;    // ����׼ȷ�ĸ��� 
		}
		double rate = num_correct*100.0 / num_new ;  // ����׼ȷ�� 
		// printf("%d num_correct=%d  rate= %f\n", k,num_correct, rate );
		// ���㲻ͬKֵ�У�ʹ��׼ȷ������Kֵ 
		if( rate > max_rate )
		{
			max_rate = rate ;   // ���׼ȷ�� 
			max_rate_k = k ;    // ׼ȷ������Kֵ 
		}
	}
	// cout <<" **max_k" <<max_rate_k <<"  max_rate " <<max_rate <<endl ;	
	ofile_new.close() ;
	/* FINISHED KNN ������� */
}

