//Copyright github.com/charmingjohn

#include "windows.h"


//*** Skeleton CODE for LBP + SVM Classifier ***//
// 19.01.29 JYJ @ CAPP
// OPENCV 3 ver
//============================================//
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <opencv2/ml/ml.hpp>
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
#include <vector>
#include <queue>
using namespace cv;
using namespace cv::ml;

// 일단 100개만 학습해보자.
//#define TRAIN_MAX	500

int loadImageAndLBP(char* img_path, std::vector<Mat>& outputDescVec, std::vector<int>& outputLabelVec, int label);
int loadAndComputeImages(char* home_dir, std::vector<Mat>& outputDescVec, std::vector<int>& outputLabelVec);
Mat get_LBP_desc(Mat inputImg);

// 주어진 폴더의 파일 목록을 가져온다. 윈도우에 특화된 함수이다.
int get_file_list(char* home_dir, std::vector<std::string> &strs)
{
	TCHAR thome[MAX_PATH];
	MultiByteToWideChar(CP_ACP, 0, home_dir, strlen(home_dir) + 1, thome, MAX_PATH);

	TCHAR tsrch[MAX_PATH];
	wsprintf(tsrch, L"%s\\*.*", thome);

	int fcount = 0;
	WIN32_FIND_DATA wfd;
//	TCHAR tpath[MAX_PATH];
	HANDLE hfind = FindFirstFile(tsrch, &wfd);
	while (hfind != INVALID_HANDLE_VALUE) {
		if (!(wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) { // 서브 폴더나 ., ..가 아닌 경우에
			TCHAR fpath[MAX_PATH];
			wsprintf(fpath, L"%s\\%s", thome, wfd.cFileName);
			char szpath[MAX_PATH];
			WideCharToMultiByte(CP_ACP, 0, fpath, lstrlen(fpath) + 1, szpath, MAX_PATH, NULL, NULL);
			std::string std_str(szpath);
			strs.push_back(std_str);
		}
		fcount++;
		if (!FindNextFile(hfind, &wfd)) {
			FindClose(hfind);
			break;
		}
	}
	return fcount;
}


// 데이터 이미지를 train과 test로 나눈다.
void lbp_svm_prep_dataset(char* train_text_name, char* posi_dir, char* nega_dir)
{
	//	char train_text_name[MAX_PATH];
	//	sprintf_s(train_text_name, MAX_PATH, "%s\\train.txt", home_dir);
	FILE* f = NULL;

	std::vector<std::string> feed_strings;
	std::vector<std::string> neg_strings;
	get_file_list(posi_dir, feed_strings);
	get_file_list(nega_dir, neg_strings);

	f = NULL;
	fopen_s(&f, train_text_name, "wt");
	if (f != NULL) {
		for (int i = 0; i < (feed_strings.size()*1)/4; i++)
			fprintf(f, "%s,1\n", feed_strings[i].c_str()); // DogFeed = 1
		for (int i = 0; i < (neg_strings.size()*1)/4; i++)
			fprintf(f, "%s,-1\n", neg_strings[i].c_str()); // DogNeg = -1
	}
	fclose(f);

	std::cout << "Positive train data set = " << feed_strings.size() << std::endl;
	std::cout << "Negative train data set = " << neg_strings.size() << std::endl;
}

void lbp_svm_train(char* svm_xml_path, char* train_posi , char* train_nega )
{
	// 데이터 이미지를 train과 test로 나눈다.
	char* train_txt_path = "C:\\temp\\lbp_test\\train.txt";
	lbp_svm_prep_dataset(train_txt_path, train_posi, train_nega);

	// 1: Read Image/Label Files and Compute LBP Descriptor
	std::cout << " Reading Input and Compute LBP Descriptor " << std::endl;
	Mat ImgDesc; // 이미지 descriptor
	std::vector<Mat> ImgDescVec; // 이미지 descriptor Vector ex) ImgDescVec[i] : i번째 Image의 Descriptor
	std::vector<int> ImgLabels; // 이미지들의 label들 ex) ImgLabels[i] : i번째 Image의 Label
	// TRAIN용 LBP를 생성한다
	int num_of_data = loadAndComputeImages(train_txt_path, ImgDescVec, ImgLabels); // ImgDesc와 ImgDescVec, ImgLabel들을 update
	std::cout << " Finished Reading Input and Compute LBP Descriptor " << std::endl;

	// 2: Convert Descriptors&Labeling datum for SVM format
	Mat trainData; // SVM에 넣어주기 위한 format. Training Image의 Descriptor를 push_back으로 하나의 Mat에 넣어줌
	Mat trainDataLabel;  // SVM에 넣어주기 위한 format. Training Image의 Label을 ImageDesc와 같은 순서로 넣어줌
	std::cout << " Make Training Datum " << std::endl;
	for (int i = 0; i < num_of_data; i++) {
		trainData.push_back(ImgDescVec[i]);
		//		trainDataLabel.push_back(Mat(1, 1, CV_32SC1, ImgLabels[i]));
		trainDataLabel.push_back(Mat(1, 1, CV_32SC1, ImgLabels[i]));
	}
	Ptr<TrainData> trainDataPt = TrainData::create(trainData, ROW_SAMPLE, trainDataLabel);

	// 3: Set SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);

	svm->setKernel(SVM::LINEAR);

	//svm->setKernel(SVM::POLY);
	//svm->setDegree(5.0);

	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));

	// 4: Train SVM
	std::cout << " Train SVM " << std::endl;
	svm->trainAuto(trainDataPt);

	// 5: Save SVM
	std::cout << " Save SVM model " << std::endl;
	svm->save(svm_xml_path);

//	delete svm;
}

/*
void sift_svm_validate(char* home_dir, std::string answerFileName)
{
	// 1 : Read Answer File
	std::vector<int> answer;
	std::ifstream labelFilePath(answerFileName);
	while(!labelFilePath.eof()){
		std::string line;
		getline(labelFilePath, line);
		int label = atoi(line.c_str());
		answer.push_back(label);
	}
	
	// 2: Read Test Files and Compute LBP Descriptors
	Mat TestImgDesc;
	std::vector<Mat> TestImgDescVec;
	std::cout<<" Reading Input and Compute Descriptor "<<std::endl;
	loadAndComputeImages(home_dir, TestImgDesc, TestImgDescVec); // Write Your Own Code
	
	// 3: Load Trained SVM
	std::cout<<" Load Model "<<std::endl;
	Ptr<SVM> svm = SVM::load("svm.xml");

	// 4: predict (example)
	for(int i =0; i<num_of_data; i++){
		if(svm->predict(TestImgDescVec[i]) == answer[i]){
			std::cout<<" Correct ! "<<std::endl;
		}
	}

	// 5: calculate Precision/Recall of each classes
	// Write Your Own Code Here----


	///-----------------------------
}
*/

// 학습용이든, 테스트용이든 LBP Descriptor계산하는 함수
int loadAndComputeImages(char* text_path, std::vector<Mat>& outputDescVec, std::vector<int>& outputLabelVec)
{
	// TO DO:
	// read image and compute descriptor
	// store it to the outputDesc & outputDescVec & outputLabelVec

	// 1 : Read Image path
	char szbuf[200];
	int count = 0;
	FILE * inputFiles = NULL;
	fopen_s(&inputFiles, text_path, "r");
	if (inputFiles != NULL) {
		while (fgets(szbuf, sizeof(szbuf), inputFiles) != NULL) {
			char fpath[200];
			char* cpos = strchr(szbuf, ','); // ',' 위치를 구한다.
			if (cpos != NULL) {
				*cpos = 0;
				strcpy_s(fpath, szbuf);
				int label = atol(cpos + 1); // cpos 다음 자리가 라벨 숫자이다.

				Mat img, img_gray;
				img = imread(fpath); // Read Images
				cvtColor(img, img_gray, CV_BGR2GRAY);
				Mat descriptors = get_LBP_desc(img_gray); // Write Your Own code
														  //outputDesc.push_back(descriptors);
				outputDescVec.push_back(descriptors);
				outputLabelVec.push_back(label);

				count++;
			}
		}
		fclose(inputFiles);
	}

	return count;
}

/* 참고용으로 남겨둡니다...
	!feof(inputFiles)){
			fscanf(inputFiles, "%s\n", &path);
			Imgpath.push(path);
			count++;
		}
	}
	int num_of_data= count;
	fclose(inputFiles);	
	std::queue<std::string> Imgpath; //Image 경로 저장하는 queue

	Imgpath.pop();

	// 2: Update ImgVec, ImgDescVec, Label Vec
	Mat img, img_gray;
	int data_per_class = num_of_data / nclasses;
	for(int i=0; i<num_of_data; i++){
		
		std::string path_ = Imgpath.front();
	}
}
*/

#if 0 // OpenCV SVM example...
int main(int, char**)
{
	// Set up training data
	int labels[4] = { 1, -1, -1, -1 };
	float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32F, trainingData);
	Mat labelsMat(4, 1, CV_32SC1, labels);
	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	// Show the decision regions given by the SVM
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);
			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
		}
	}
	// Show the training data
	int thickness = -1;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness);
	// Show support vectors
	thickness = 2;
	Mat sv = svm->getUncompressedSupportVectors();
	for (int i = 0; i < sv.rows; i++)
	{
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness);
	}
	imwrite("result.png", image);        // save the image
	imshow("SVM Simple Example", image); // show it to the user
	waitKey();
	return 0;
}
#endif
