/*####################################################################################################################*/
//
//	Name:	Rahul Maheshwari
//	Assignment.h : Header file
//      Assignment.cpp : Class file
//	Source.cpp : This program performs mean shift algorithm and watershed algorithm using openCV library. 
//
/*####################################################################################################################*/


// include necessary header files
#include "Assignment2.h"

int main( int argc, char** argv )
{
	Mat input1, input2; // create Image container in opencv
	
	input1 = imread("bookshelf_1.jpg", 1);  // read an input image
	input2 = imread("bookshelf_2.jpg", 1);  // read an input image
	if(!input1.data || !input2.data)    // check whether reading is successful or not
	{
		cout << "No image data" << endl;
		return -1;
	}

	Assignment2 m1(input1);  // create object of class
	m1.extractSiftFeatures(); // extractSIFTFeature for that object
	
	Assignment2 m2(input2); //create object of class
	m2.extractSiftFeatures(); // extractSIFTFeature for that object
	
	m1.displayFeatures(m2);  // display both the feature extracted image together
	
	vector< DMatch > good_matches;
	good_matches = m1.FindMatchesEuclidian(m2);
	m1.displayGoodMatches(m2, good_matches);   // drawMatches using function created for class
	
	imshow("Image1_SIFT", m1.getSIFTImage());
	imshow("Image1_SIFT", m1.getSIFTImage()); // display images

	//Mat H = m1.computeRANSAC_opencv(m2);
	//cout << H << endl;
	
	
	Mat H = m1.computeRANSAC(m2);  // compute H without using openCV function
	//cout << H << endl;
	
	Mat output = m1.warpImage(H);  // get the effect of H on input image
	imshow("output", output);

	Mat overlayingoutput = m1.displayOverlaying(m2, H, input1);   // compute overlayed outputs
	imshow("Overlaying output", overlayingoutput);   // dislay overlayed output
	 
	waitKey(0);  // wait for stroke key

	cout << "End of Program: " << endl;
	return 0;
}
