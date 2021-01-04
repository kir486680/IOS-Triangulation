#include "LaneDetector.hpp"

using namespace cv;
using namespace std;

float x[3][3]= {{1.15777818e+03, 0.00000000e+00, 6.67113857e+02},
    {0.00000000e+00, 1.15282217e+03, 3.86124583e+02},
    {0.00000000e+00, 0.00000000e+00, 1.00000000e+00}};

Mat LaneDetector::detect_lane(Mat image1, Mat image2) {
    
    //vector<KeyPoint> keypoints;
    //Mat desc;
    //Ptr<ORB> detector = ORB::create();
    //detector->detectAndCompute(image1, Mat(), keypoints, desc);
    
    //Ptr<DescriptorExtractor> descriptor = ORB::create();
    //vector<KeyPoint> keypoints_1 ={KeyPoint(512, 512, 0)};
    //Mat descriptors_1;
    //descriptor->compute (image1, keypoints_1, descriptors_1 );
    //vector<KeyPoint> keypoints_2 ={KeyPoint(512, 512, 0)};
    //Mat descriptors_2;
    //descriptor->compute (image2, keypoints_2, descriptors_2 );
    
    cout<<image1.size();
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    detector->detect ( image1,keypoints_1 );
    detector->detect ( image2,keypoints_2 );
    
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( image1, keypoints_1, descriptors_1 );
    descriptor->compute ( image2, keypoints_2, descriptors_2 );
    
    BFMatcher matcher(NORM_L2);
    std::vector<vector<DMatch> > knn_matches;
    matcher.knnMatch(descriptors_2, descriptors_2, knn_matches,2);
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    Mat img_matches;
    drawMatches( image1, keypoints_1, image2, keypoints_2, good_matches, img_matches, Scalar::all(-1),Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    //cout<<descriptors_2;
    
    //drawMatches(image1, keypoints_1, image2, keypoints_2, match1, img_matches1);

    
    
    return img_matches;
}
