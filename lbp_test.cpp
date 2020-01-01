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
int loadImagesAndLabels(char* img_path, std::vector<std::string>& outputDescVec, std::vector<int>& outputLabelVec, int label);
int loadAndComputeImages(char* home_dir, std::vector<Mat>& outputDescVec, std::vector<int>& outputLabelVec);
Mat get_LBP_desc(Mat inputImg);
void copy_image(const char* src, const char* dd);

void lbp_svm_test(char* svm_xml_path, 	char* posi_folder, char* nega_folder )
{
	std::cout << "Strating test\r\n";

	CreateDirectoryA("C:\\temp\\lbp_test", NULL);
	CreateDirectoryA("C:\\temp\\lbp_test\\tt", NULL);
	CreateDirectoryA("C:\\temp\\lbp_test\\tn", NULL);
	CreateDirectoryA("C:\\temp\\lbp_test\\nt", NULL);
	CreateDirectoryA("C:\\temp\\lbp_test\\nn", NULL);

	// 5: Loed SVM
	std::cout << " Load SVM model " << std::endl;
	Ptr<SVM> svm = Algorithm::load<SVM>(svm_xml_path);

	// test set load
	std::vector<std::string> feed_strings;
	//std::vector<Mat> TestImgDescVec; // 이미지 descriptor Vector ex) ImgDescVec[i] : i번째 Image의 Descriptor
	std::vector<int> TestImgLabels; // 이미지들의 label들 ex) ImgLabels[i] : i번째 Image의 Label
	int posi_count = loadImagesAndLabels(posi_folder, feed_strings, TestImgLabels, 1);
	int nega_count = loadImagesAndLabels(nega_folder, feed_strings, TestImgLabels, -1);
	int full_count = posi_count + nega_count;

	// 6: predict (example)
	int correct_count = 0;
	int tt_count = 0;
	int tn_count = 0;
	int nt_count = 0;
	int nn_count = 0;

	for (int i = 0; i < full_count; i++) {
		std::string fname = feed_strings[i].c_str();
		Mat img, img_gray;
		img = imread(fname.c_str()); // Read Images
		cvtColor(img, img_gray, CV_BGR2GRAY);

		Mat descriptors = get_LBP_desc(img_gray); // Write Your Own code

		float result = svm->predict(descriptors);
		if (result == TestImgLabels[i]) {
			std::cout << " Correct ! " << std::endl;
			correct_count++;
		}
		else {
			std::cout << " Incorrect ! " << std::endl;
		}

		if (TestImgLabels[i] == 1) {
			if (result == 1) {
				tt_count++;
				copy_image(fname.c_str(), "tt");
			}
			else {
				tn_count++;
				copy_image(fname.c_str(), "tn");
			}
		}
		else {
			if (result == 1) {
				nt_count++;
				copy_image(fname.c_str(), "nt");
			}
			else {
				nn_count++;
				copy_image(fname.c_str(), "nn");
			}
		}
	}
	std::cout<< "correction=" << correct_count << " total=" << full_count << "\n"; 
	std::cout << "accuracy =" << ((float)correct_count/(float)full_count) * 100.0 << "\n";
	std::cout << "Label=T, Predict=P count = " << tt_count << "\n";
	std::cout << "Label=T, Predict=N count = " << tn_count << "\n";
	std::cout << "Label=F, Predict=P count = " << nt_count << "\n";
	std::cout << "Label=F, Predict=N count = " << nn_count << "\n";

	getchar(); // 윈도우니까 창이 닫히지 않게 배려해준다. 착한 연구자.
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

int IsImageFile(char* fn)
{
	char uppername[MAX_PATH];
	int  upperlen = 0;
	while (*fn) {
		uppername[upperlen++] = toupper(*fn);
		fn++;
	}
	uppername[upperlen] = 0;
	if (strcmp(uppername + upperlen - 4, ".JPG") == 0)
		return 1;
	if (strcmp(uppername + upperlen - 4, ".PNG") == 0)
		return 1;
	if (strcmp(uppername + upperlen - 4, ".JPEG") == 0)
		return 1;

	return 0;
}

// 학습용이든, 테스트용이든 LBP Descriptor계산하는 함수
// 폴더의 이미지를 읽어서 LBP를 계산해서 outputDescVec에 넣어준다.
// 전달받은 label 값을 outputLabelVec에 넣어준다.
int loadImageAndLBP(char* folder, std::vector<Mat>& outputDescVec, std::vector<int>& outputLabelVec, int label)
{
	int count = 0;

	// 주어긴 폴더의 이미지 파일을 읽어서 작업한다.
	WIN32_FIND_DATAA wfd;
	char filter[MAX_PATH];
	sprintf_s(filter, MAX_PATH, "%s\\*.*", folder);
	HANDLE hFind = FindFirstFileA(filter, &wfd);
	while (hFind != INVALID_HANDLE_VALUE) {
		if (!(wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && IsImageFile(wfd.cFileName)) {
			// read image and compute descriptor
			// store it to the outputDesc & outputDescVec & outputLabelVec
			char imgpath[MAX_PATH];
			sprintf_s(imgpath, "%s\\%s", folder, wfd.cFileName);
			Mat img, img_gray;
			img = imread(imgpath); // Read Images
			cvtColor(img, img_gray, CV_BGR2GRAY);

			Mat descriptors;
			descriptors = get_LBP_desc(img_gray); // Write Your Own code
															  //outputDesc.push_back(descriptors);
			outputDescVec.push_back(descriptors);
			outputLabelVec.push_back(label);

			count++;
		}
		if (!FindNextFileA(hFind, &wfd)) {
			FindClose(hFind);
			break;
		}
	}

	return count;
}

int loadImagesAndLabels(char* folder, std::vector<std::string>& outputDescVec, std::vector<int>& outputLabelVec, int label)
{
	int count = 0;

	// 주어긴 폴더의 이미지 파일을 읽어서 작업한다.
	WIN32_FIND_DATAA wfd;
	char filter[MAX_PATH];
	sprintf_s(filter, MAX_PATH, "%s\\*.*", folder);
	HANDLE hFind = FindFirstFileA(filter, &wfd);
	while (hFind != INVALID_HANDLE_VALUE) {
		if (!(wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && IsImageFile(wfd.cFileName)) {
			// read image and compute descriptor
			// store it to the outputDesc & outputDescVec & outputLabelVec
			char imgpath[MAX_PATH];
			sprintf_s(imgpath, "%s\\%s", folder, wfd.cFileName);
			outputDescVec.push_back(imgpath);
			outputLabelVec.push_back(label);

			count++;
		}
		if (!FindNextFileA(hFind, &wfd)) {
			FindClose(hFind);
			break;
		}
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

#define	DO_EQ	0

Mat get_LBP_desc(Mat src)
{
	Mat img;
	if (DO_EQ) {
		/// Apply Histogram Equalization
		Mat dst;
		equalizeHist(src, dst);
		img = dst;
	}
	else {
		img = src;
	}

	int rowsize = img.cols;
	// Write your own code Here--
	unsigned char *bmat = (unsigned char*)malloc(img.rows * img.cols);
	for (int r = 0; r<img.rows;r++) {
		for (int c = 0; c<img.cols; c++) {
			bmat[r*img.cols + c] = img.at<uchar>(r, c);
		}
	}

#define GET_RC(m,r,c,rs) ((m)[(r)*(rs)+(c)])

	Mat dst = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1;i<img.rows - 1;i++) {
		for (int j = 1;j<img.cols - 1;j++) {
			uchar center = GET_RC(bmat, i, j, rowsize);
			unsigned char code = 0;
			if (GET_RC(bmat, i - 1, j - 1, rowsize) > center)
				code |= 0x80;
			if (GET_RC(bmat, i - 1, j, rowsize) > center)
				code |= 0x40;
			if (GET_RC(bmat, i - 1, j + 1, rowsize) > center)
				code |= 0x20;
			if (GET_RC(bmat, i, j + 1, rowsize) > center)
				code |= 0x10;
			if (GET_RC(bmat, i + 1, j + 1, rowsize) > center)
				code |= 0x08;
			if (GET_RC(bmat, i + 1, j, rowsize) > center)
				code |= 0x04;
			if (GET_RC(bmat, i + 1, j - 1, rowsize) > center)
				code |= 0x02;
			if (GET_RC(bmat, i, j - 1, rowsize) > center)
				code |= 0x01;
			dst.at<uchar>(i - 1, j - 1) = code;
		}
	}
	free(bmat);

	// 히스토그램을 만들자.
	Mat hist = Mat::zeros(1, 256, CV_32F);
	for (int r = 0;r < dst.rows; r++) {
		for (int c = 0; c < dst.cols; c++) {
			uchar center = dst.at<uchar>(r, c);
			float fv = hist.at<float>(0, center);
			fv += 1.0;
			hist.at<float>(0, center) = fv;
		}
	}

	
	// 히스토그램 normalization
	Mat result = Mat::zeros(1, 256, CV_32F);
	normalize(hist, result, 0, 65535, NORM_MINMAX);

//	FileStorage fs("histogramresult_%d.txt", FileStorage::WRITE);
//	fs << "mat" << hist;
//	fs.release();

	return result;
}

void copy_image(const char* src, const char* dd)
{
	int len = strlen(src);
	while (len > 0) {
		if (src[len] == '\\')
			break;
		len--;
	}
	char dstpath[MAX_PATH];
	sprintf_s(dstpath, MAX_PATH, "c:\\temp\\lbp_test\\%s\\%s", dd, src + len);
	CopyFileA(src, dstpath, FALSE);
}


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
