//Copyright github.com/charmingjohn

#include <opencv2/core/core.hpp>  
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>
#include "windows.h"

using namespace cv;
using namespace std;

void lbp_svm_train(char* svm_xml_path, char* train_posi, char* train_nega);
void lbp_svm_test(char* svm_xml_path, char* posi_folder, char* nega_folder);

int main(int argc, char** argv)
{
	char* lbp_svm_path = "C:\\temp\\lbp_test\\svm.xml";
	DeleteFileA(lbp_svm_path);

	char* train_posi = "C:\\temp\\lbp_test\\dogfeed_cropped";
	char* train_nega = "C:\\temp\\lbp_test\\negative_cropped";
	lbp_svm_train(lbp_svm_path, train_posi, train_nega);

	char* posi_folder = "C:\\temp\\lbp_test\\dogfeed_test";
	char* nega_folder = "C:\\temp\\lbp_test\\negative_test";
	lbp_svm_test(lbp_svm_path, posi_folder, nega_folder);

	return 0;
}