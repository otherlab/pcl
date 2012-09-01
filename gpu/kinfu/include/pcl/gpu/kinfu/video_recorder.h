/*
00002  * Software License Agreement (BSD License)
00003  *
00004  *  Copyright (c) 2011, Willow Garage, Inc.
00005  *  All rights reserved.
00006  *
00007  *  Redistribution and use in source and binary forms, with or without
00008  *  modification, are permitted provided that the following conditions
00009  *  are met:
00010  *
00011  *   * Redistributions of source code must retain the above copyright
00012  *     notice, this list of conditions and the following disclaimer.
00013  *   * Redistributions in binary form must reproduce the above
00014  *     copyright notice, this list of conditions and the following
00015  *     disclaimer in the documentation and/or other materials provided
00016  *     with the distribution.
00017  *   * Neither the name of Willow Garage, Inc. nor the names of its
00018  *     contributors may be used to endorse or promote products derived
00019  *     from this software without specific prior written permission.
00020  *
00021  *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
00022  *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
00023  *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
00024  *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
00025  *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
00026  *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
00027  *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
00028  *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
00029  *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
00030  *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
00031  *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
00032  *  POSSIBILITY OF SUCH DAMAGE.
00033  *
00034  *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
00035  */

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/gpu/kinfu/kinfu.h>

//using namespace cv;

#define PCL_FOURCC(c1, c2, c3, c4) (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)

using namespace pcl;
using namespace pcl::gpu;

namespace pcl
{
    class PCL_EXPORTS BufferedRecorder
    {
    public:
        typedef cv::Mat Mat;
        typedef cv::Size Size;

        int total_, total_kinfu;

        std::vector<cv::Mat> images_;

        std::vector< DeviceArray2D<PixelRGB> > kinfu_images;

        const static int FOURCC = PCL_FOURCC ('X', 'V', 'I', 'D');
        //const static int FOURCC = PCL_FOURCC('M', 'J', 'P', 'G');
        //const static int FOURCC = -1; //select dialog

        BufferedRecorder(const Size size = Size (1280, 1024), size_t frames = 30 * 20) : size_ (size), total_ (0), total_kinfu(0)
        {
            images_.resize (frames);
//            depths_.resize (frames);
//            views_.resize (frames);


            for (size_t i = 0; i < frames; ++i)
            {
                images_[i].create (size, CV_8UC3);
//                depths_[i].create (size, CV_16U);
//                views_[i].create (size, CV_8UC3);
            }
        }

        void
        push_back (const cv::Mat& image)//, const cv::Mat& depth, const cv::Mat& view)
        {
            if (total_ < images_.size ())
            {
                image.copyTo (images_[total_]);
//                depth.copyTo (depths_[total_]);
//                view.copyTo ( views_[total_]);
            }
            else
            {
                images_.push_back (image.clone ());
//                depths_.push_back (depth.clone ());
//                views_.push_back ( view.clone ());
            }
            ++total_;
        }
        void
        push_back_kinfu_image (DeviceArray2D<PixelRGB> img)
        {
            if(total_kinfu >= 600)
                return;

            kinfu_images.push_back(img);

            ++total_kinfu;
        }

        bool
        save (const std::string& file = "outputs/video.avi") const
        {
            std::vector<int> compression_params;
                compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(0);

            if (total_ > 0)
            {
//                Size sz (size_.width * 3, size_.height);
                Size sz (size_.width, size_.height);
                cv::VideoWriter vw (file, FOURCC, 20, sz, true);



                if (vw.isOpened () == false)
                    return std::cout << "Can't open video for writing" << std::endl, 0;

                std::cout << "Encoding";
                cv::Mat all (sz, CV_8UC3, cv::Scalar (0));

                for (size_t k = 0; k < total_; ++k)
                {

                    cv::Mat t;
//                    cv::Mat d = depths_[k];
                    cv::Mat i = images_[k];
//                    cv::Mat v = views_[k];




                    int pos = 0;
//                    t = all.colRange (pos, pos + d.cols);
//                    cv::cvtColor (d, t, CV_GRAY2BGR);
//                    pos += d.cols;

                    t = all.colRange (pos, pos + i.cols);
                    i.copyTo (t);
                    pos += i.cols;

//                    t = all.colRange (pos, pos + v.cols);
//                    v.copyTo (t);
//                    pos += v.cols;

                    vw << all;

                    //sema

                    //write into an image file
                    char fn[50];
                    sprintf(fn, "outputs/rgb/rgb_%04d.png", (int)k);
                    try {
                        cv::imwrite(fn, i, compression_params);
                    }
                    catch (std::runtime_error& ex) {
                        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
                        return 1;
                    }

                    std::cout << ".";
                }
                std::cout << "Done" << std::endl;
            }
            return true;
        }
        void
        reset()
        {
            total_ = 0;
            total_kinfu = 0;
            images_.clear();
            kinfu_images.clear();
        }

    private:

//        std::vector<cv::Mat> depths_;
//        std::vector<cv::Mat> views_;
        Size size_;

    };
}
