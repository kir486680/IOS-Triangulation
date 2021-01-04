//
//  LangeDetectorBridge.m
//  SimpleLaneDetection
//
//  Created by Anurag Ajwani on 28/04/2019.
//  Copyright © 2019 Anurag Ajwani. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <Foundation/Foundation.h>
#import "LaneDetectorBridge.h"
#include "LaneDetector.hpp"

@implementation LaneDetectorBridge
    
- (UIImage *) detectLaneIn: (NSArray *) image {
    cv::Mat opencvImage;
    cv::Mat opencvImage1;
    cv::Mat opencvImage2;
    UIImageToMat(image[0], opencvImage1, true);
    UIImageToMat(image[1], opencvImage2, true);
    LaneDetector featureDetector;
    opencvImage = featureDetector.detect_lane(opencvImage1, opencvImage2);
    return MatToUIImage(opencvImage);
}
    
    @end
