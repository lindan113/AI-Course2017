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

double alpha = 0.6 ;
vector<string> word;    //��training���� 
vector<Passage> training ; //ѵ�������������� 
double emotion[6] ;  // ÿ����еĸ��� = ��ǩΪEi����������/�������� 
int Times_training[PASSAGE_NUM][WORD_NUM] ;  	// ѵ����ÿƪ����i�ĵ���j���ִ����� 
double Probability[6][WORD_NUM] ; //6�������ÿ�����ʷֱ���ֵ��ܴ���
int all_sum[6] ;			// ÿһ����а����ĵ�������
// PASSAGE_NUM 3000ѵ������������ ,WORD_NUM 60000�������� 

int main()
{
	/** ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	 
	ifstream file0;   
	//file0.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nbSMALL\\train_set.csv");
	file0.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\����ԭʼ����\\train_set.csv");

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
	
	int num_training = ( SplitVec0.size() - 1 ) / 2 ;	//ѵ������������ 	
	/** FINISHED ��training�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/
	
	/* ����ѵ�������ʴ������󣬼�����ÿƪ���¸������ʳ��ִ��� */ 
	for (int i = 0; i<num_training; i++)	
	{
		Passage p;      				//�����iƪ����  	 
		p.label = SplitVec0[2*i+1] ; 	//ÿһƪ���µ�label 
		string tmp = SplitVec0[2*i] ;       	//��iƪ����û�нض�ǰ 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����iƪ���µĵ���
		training.push_back( p ) ;		//��iƪ���²���training����vector 

		//�����i�е�Times���� 
		for( int j=0; j<p.voc.size(); j++ ) // �鿴������ÿ������ 
		{
			string subtmp = p.voc[j] ;	   // ��iƪ���µĵ�j������ 
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 
			
			if( iterloc!= word.end() )  //���ʱ��Ѿ������������ 
			{
				int loc = iterloc-word.begin() ;  	// ��j���������ڵ��ʿ�λ�� 
				Times_training[i][loc] ++;      // ����+1 
			}
			else //�µ��� 
			{	
				word.push_back( subtmp ) ;			// �����µ��� 
				Times_training[i][ word.size()-1 ] = 1; 	// ��j���������ڵ��ʿ�λ�� ��// ����=1 
			}		
		}				
	}	
	/* FINISHED ����ѵ�������ʴ������� */ 
		
	/* �����i������£����ֵ���j�ĸ��� ����prob=P(Xj,Ei)
	ÿ�����ʷֱ���6����е��ܴ�����ÿһ����г��ֵĵ���������ÿһ����г��ֵĲ��ظ�������Ŀ*/ 
	
	/* ����һ������Probability[i][j], ��i������е���j���ִ��� */ 
	string label[6]={ "anger", "disgust", "fear", "joy", "sad", "surprise" } ;	
	// �����ȡֵ�� 6�����
	for( int i=0; i<6; i++)		//��i����б�ǩ 
	{	// ����ѵ�������ҳ���ǩΪEi������ 
		for( int t=0; t<num_training; t++)  //ѵ�����еĵ�tƪ���� 
		{	// vector<Passage> training ��ѵ�������������� 
			if( training[t].label == label[i] )  
			{	// �����tƪ���� ���� ��i�����
				emotion[i] ++ ;  // ���ΪEi����������+1 				
				for( int j=0; j<word.size(); j++)  //�������е��� 
				{	// Times_training[t][j] ��ѵ������tƪ���µĵ���j���ִ���  					
					Probability[i][j] += Times_training[t][j] ;
				}	
			}
		}						
	}  /* FINISHED  �������Probability[i][j] */  
	
	/* ����һ������all_sum[i], ��i������е��ʳ����ܴ������ظ���Ҳ�㣩 */  
	for( int i=0; i<6; i++)
	{
		for( int j=0; j<word.size(); j++)
		{	// �ۼӵ���j�ڵ�i������г��ִ���		
			all_sum[i] += Probability[i][j] ; 
		}
	} /* FINISHED ����һ������all_sum[i] */ 
	
	/*Ϊ�˼��㷽�㣬�� Probability[i][j]��i������е���j�����ܴ���
	���ĳɡ� P(Xj|Ei) */
	for( int i=0; i<6; i++)
	{
		for( int j=0; j<word.size(); j++)
		{   // ������˹ƽ����ʽ
			Probability[i][j] = (Probability[i][j]+alpha) / ( all_sum[i] + alpha*word.size()) ;	
		}
	}

	/** ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/ 	   
	ifstream file1; 
	//file1.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\nbSMALL\\test_set.csv");
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
	int num_new = ( SplitVec1.size() - 1 ) / 2 ;	//validation�������� 	
	/** FINISHED ��validation�ļ�����ȡ�ļ����ָ�Ϊ�ı��ͱ�ǩ **/
	
	
	/* * * * * * * * * * * * * * * ������ * * * * * * * * * * * * * * * * */ 
	ofstream ofile_new;               //��������ļ�OneHot.txt
	//ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\classification_dataset\\small\\15352204_Sample_KNN_classification.csv");    //��Ϊ����ļ���	
	ofile_new.open("D:\\iѧϰ\\����\\�˹����� �����\\Lab2\\DATA\\����ԭʼ����\\15352204_LinDan_NB_classification.csv");   
	
	ofile_new <<"textid" <<"," <<"label" <<endl ;
	double num_correct = 0;
		
	/* �ָ��ʱ��NB���������� */ 
	double pro[6] ;  // ÿ����еĸ���	
	for (int x = 0; x<num_new; x++)	//���Լ��ĵ�xƪ���� 
	{
		Passage p;      				//�����xƪ����  	 
		p.label = SplitVec1[2*x+1] ; 	//ÿһƪ���µ�label 
		string tmp = SplitVec1[2*x] ;   //��xƪ����û�нض�ǰ 		
		split( p.voc, tmp , is_any_of(" "), token_compress_on ); //�����xƪ���µĵ���
		// ÿƪ���µ���и��ʳ�ʼ�� P(Ei) 
		for( int i=0; i<6; i++)
			pro[i] = 1.0 * emotion[i] / num_training;  
			
		double max_pro = 0 ; 	// ��и������ֵ 
		int max_i = 0;			// ��и������ֵ��Ӧ����±� 
		for( int j=0; j<p.voc.size(); j++ ) // �鿴������ÿ������
		{ 	
			string subtmp = p.voc[j] ;  // ��xƪ���µĵ�j������ 	   
			//�����Ƿ��Ѿ����ڵ��ʿ� 				
			vector<string>::iterator iterloc = find( word.begin(), word.end() , subtmp ) ; 			
			if( iterloc!= word.end() )  //���ʱ��Ѿ������������ 
			{
				int loc = iterloc-word.begin() ;  	//��j���������ڵ��ʿ�λ�� 
				for( int i=0; i<6; i++)
					pro[i] *= Probability[i][loc] ;
				// ����NB���๫ʽ���ۻ� P(Xj|Ei)P(Ei)
			}
			else //�µ���,ԭ������Ϊ0������������˹ƽ�� 
			{	
				for( int i=0; i<6; i++)
				{	//������˹ƽ�� 
					pro[i] = pro[i] * alpha / ( all_sum[i] + word.size()) ;
				}					
			}		
		}
				
		// ������Լ���xƪ���£���һ����и��ʸ� 
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
		{	// Ԥ����==��ȷ�𰸣� 
			num_correct ++ ; //����׼ȷ�� 
		}	
	}
	/* FINISHED */ 
	
	ofile_new.close() ;
	double rate = 1.0 * num_correct / num_new ;
	//cout <<"alpha: "<<alpha <<" num_correct: " <<num_correct <<" rate: " <<rate <<endl ;
}
	
	
	


