/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu/kinfu.h>


#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

//sema
//opencv
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//#ifdef HAVE_OPENCV
//  #include <opencv2/opencv.hpp>
//  #include <opencv2/gpu/gpu.hpp>
//  #include <pcl/gpu/utils/timers_opencv.hpp>
//#endif

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
  namespace gpu
  {
    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
  }
}

//sema
#define DESC_LENGTH 128
void opencvSIFT(cv::Mat input, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors);
void fill_video_frame(cv::Mat& video_frame, KinfuTracker::View rgb24);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SEMA
//////////
void
pcl::gpu::KinfuTracker::extractObject(pcl::ModelCoefficients::Ptr box_boundaries,
                                     pcl::ModelCoefficients::Ptr coefficients)
{
    Array3f cell_size = tsdf_volume_->getVoxelSize();
    box_boundaries->values[0] /= (cell_size[0]);
    box_boundaries->values[1] /= (cell_size[0]);
    box_boundaries->values[2] /= (cell_size[1]);
    box_boundaries->values[3] /= (cell_size[1]);
    box_boundaries->values[4] /= (cell_size[2]);
    box_boundaries->values[5] /= (cell_size[2]);

    coefficients->values[0] *= cell_size[0];
    coefficients->values[1] *= cell_size[1];
    coefficients->values[2] *= cell_size[2];

    tsdf_volume_->cleanTsdfByROIandPlane(box_boundaries, coefficients);
}

void
pcl::gpu::KinfuTracker::reduceTsdfWeights(pcl::ModelCoefficients::Ptr box_boundaries)
{
    Array3f cell_size = tsdf_volume_->getVoxelSize();
    box_boundaries->values[0] /= (cell_size[0]);
    box_boundaries->values[1] /= (cell_size[0]);
    box_boundaries->values[2] /= (cell_size[1]);
    box_boundaries->values[3] /= (cell_size[1]);
    box_boundaries->values[4] /= (cell_size[2]);
    box_boundaries->values[5] /= (cell_size[2]);

    tsdf_volume_->reduceTsdfByROI(box_boundaries);
}
//SEMA
void
pcl::gpu::KinfuTracker::setROI(pcl::ModelCoefficients::Ptr box_boundaries)
{
    Array3f cell_size = tsdf_volume_->getVoxelSize();
    roi_boundaries_->values[0] = (box_boundaries->values[0]/cell_size[0]);
    roi_boundaries_->values[1] = (box_boundaries->values[1]/cell_size[0]);
    roi_boundaries_->values[2] = (box_boundaries->values[2]/cell_size[1]);
    roi_boundaries_->values[3] = (box_boundaries->values[3]/cell_size[1]);
    roi_boundaries_->values[4] = (box_boundaries->values[4]/cell_size[2]);
    roi_boundaries_->values[5] = (box_boundaries->values[5]/cell_size[2]);

    set_ROI_selection_time();
}

void
pcl::gpu::KinfuTracker::getBasePlane ()
{
    reset_plane_coeffs();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr =
            pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    DeviceArray2D<pcl::PointXYZ> cloud_device_;

    getCurrentFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr->points, c);
    cloud_ptr->width = cloud_device_.cols ();
    cloud_ptr->height = cloud_device_.rows ();
    cloud_ptr->is_dense = false;

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud = *cloud_ptr;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud.makeShared ());
    seg.segment (*inliers, *plane_coeffs);


    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given dataset.");
        return;
    }

    cerr << "Model coefficients: " << plane_coeffs->values[0] << " "
         << plane_coeffs->values[1] << " "
         << plane_coeffs->values[2] << " "
         << plane_coeffs->values[3] << endl;

    cerr << "Model inliers: " << inliers->indices.size () << endl;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker (int rows, int cols) : rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), ishand(false), isICPfail(true)
{
  const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);

  tsdf_volume_ = TsdfVolume::Ptr( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize(volume_size);

  setDepthIntrinsics (525.f, 525.f);

  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {30, 16, 10};  //10, 5, 4
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  setIcpCorespFilteringParams (default_distThres, default_angleThres);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  //SEMA
//  cloud_all = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
  cloud_curr = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

  //SEMA
  roi_boundaries_ = ModelCoefficients::Ptr(new ModelCoefficients);
  roi_boundaries_->values.resize (6);    // We need 6 values
  roi_boundaries_->values[0] = -1;
  roi_boundaries_->values[1] = -1;
  roi_boundaries_->values[2] = -1;
  roi_boundaries_->values[3] = -1;
  roi_boundaries_->values[4] = -1;
  roi_boundaries_->values[5] = -1;

  allocateBufffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  hand.reset();

  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2 : cx;
  cy_ = (cy == -1) ? rows_/2 : cy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setInitalCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  reset ();
}

void
pcl::gpu::KinfuTracker::stabilizeCameraPosition()
{
    rmats_[global_time_ - 1] = stable_Rcam_;
    tvecs_[global_time_ - 1] = stable_tcam_;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::rows ()
{
  return (rows_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::reset()
{
  if (global_time_)
    cout << "Reset" << endl;

  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  r_inv_.clear();
  t_.clear();

  tsdf_volume_->reset();

  if (color_volume_) // color integration mode is enabled
    color_volume_->reset();

  //SEMA
  init = true;
  reset_plane_coeffs();

  //SEMA
  roi_boundaries_->values[0] = -1;
  roi_boundaries_->values[1] = -1;
  roi_boundaries_->values[2] = -1;
  roi_boundaries_->values[3] = -1;
  roi_boundaries_->values[4] = -1;
  roi_boundaries_->values[5] = -1;

  vmaps_.clear();
  nmaps_.clear();

  depth_images.clear();
  depth_raw_images.clear();

  roi_selection_time = -1;

//  hand.reset();

  setScaleFactor(64.);
}

//SEMA
void
pcl::gpu::KinfuTracker::reset_plane_coeffs ()
{
    plane_coeffs = ModelCoefficients::Ptr(new ModelCoefficients);
    plane_coeffs->values.resize (4);    // We need 4 values
    plane_coeffs->values[0] = 0;
    plane_coeffs->values[1] = 1;
    plane_coeffs->values[2] = 0;
    plane_coeffs->values[3] = 1000;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::allocateBufffers (int rows, int cols)
{
  depths_curr_.resize (LEVELS);
  vmaps_g_curr_.resize (LEVELS);
  nmaps_g_curr_.resize (LEVELS);

  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

    coresps_[i].create (pyr_rows, pyr_cols);
  }
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);


  raycast_view.create(rows, cols);
//  //sema
//  vmap_save.create (rows*3, cols);
//  nmap_save.create (rows*3, cols);
}

void
pcl::gpu::KinfuTracker::init_rmats_icp()
{
    rmats_icp.operator =(rmats_);
    tvecs_icp.operator =(tvecs_);

    rmats_icp_nl.operator =(rmats_);
    tvecs_icp_nl.operator =(tvecs_);
}

void
pcl::gpu::KinfuTracker::revert_rmats_()
{
    rmats_.operator =(rmats_icp);
    tvecs_.operator =(tvecs_icp);
}

bool
pcl::gpu::KinfuTracker::reconstructWithModelProxy_NonLinearICP (PointCloud<PointXYZ>::Ptr cloud, int ctrl)
{
    if(global_time_ < 1)
        return (false);

    if(cloud->points.size() < 100 )
        return false;



    cout<<"global time: "<<global_time_<<endl;
    cout<<"rmats_: "<<rmats_.size()<<endl;

    //clear voxel representation
    tsdf_volume_->reset();

    device::Intr intr (fx_, fy_, cx_, cy_);


    bool roi_isset = false;


    std::vector<int> indices;
    cloud->is_dense = false;
    cout<<cloud->points.size()<<endl;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    cout<<cloud->points.size()<<" "<< indices.size() <<endl<<endl;

    pcl::PointCloud<pcl::PointNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointNormal>);
    tgt->is_dense = false;
    pcl::copyPointCloud(*cloud, *tgt);



    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
    norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
    //Set the number of k nearest neighbors to use for the feature estimation.
    norm_est.setKSearch (40);
    norm_est.setInputCloud (tgt);
    norm_est.compute (*tgt);

    //set point-to-plane distance metric
    pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> icp;
    typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
    if(ctrl == 1) {
        boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
        icp.setTransformationEstimation(point_to_plane);
    }
    icp.setInputTarget(tgt);

    //      icp.setRANSACOutlierRejectionThreshold(0.1);
//    icp.setRANSACIterations(100);
    icp.setTransformationEpsilon(1e-6);

    icp.setMaxCorrespondenceDistance (0.02); // in meters
    icp.setMaximumIterations(100);


     pcl::PointCloud<pcl::PointNormal> all_cloud;// = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);;

     int conv=0;
    for(int process_t = 0; process_t < global_time_; process_t++)
    {
        if(process_t == roi_selection_time)
            roi_isset = true;

        cout<<process_t<<endl;
        DepthMap depth_raw; //raw depth image
        depth_raw.upload(depth_images.at(process_t), cols());

        //get current frame vmap, nmap
        {
            device::bilateralFilter (depth_raw, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(),  scale_factor);

            if (max_icp_distance_ > 0)
                device::truncateDepth(depths_curr_[0], max_icp_distance_, scale_factor);

            for (int i = 1; i < LEVELS; ++i)
              device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

            for (int i = 0; i < LEVELS; ++i)
            {
              device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
              computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
            }
            pcl::device::sync ();
        }

        //////////////////////////////////////////////////
        // ICP
        /////////////////////////////////////////////////
        Matrix3frm Rprev = rmats_[process_t]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   tprev = tvecs_[process_t]; //  tranfrom from camera to global coo space for (i)th camera pose

        Matrix3frm Rcurr = Rprev;
        Vector3f   tcurr = tprev;

        DeviceArray2D<PointXYZ> cloud_device_;
        cloud_device_.create (rows_, cols_);
        DeviceArray2D<float4>& cd = (DeviceArray2D<float4>&)cloud_device_;
        device::convert (vmaps_curr_[0], cd);

        int c;
        cloud_device_.download (cloud_curr->points, c);
        cloud_curr->width = cloud_device_.cols ();
        cloud_curr->height = cloud_device_.rows ();
        cloud_curr->is_dense = false;


        if(cloud_curr->points.size()>0)
        {
            pcl::PointCloud<pcl::PointNormal>::Ptr src(new pcl::PointCloud<pcl::PointNormal>);
            pcl::copyPointCloud(*cloud_curr, *src);
            src->width = cloud_device_.cols ();
            src->height = cloud_device_.rows ();
            src->is_dense = false;

cout<<src->points.size()<<endl;
            //remove NAN points from the cloud
            pcl::removeNaNFromPointCloud(*src,*src, indices);

            icp.setInputCloud(src);

            cout<<src->points.size()<<endl;
            cout<<tgt->points.size()<<endl;

            pcl::PointCloud<pcl::PointNormal> Final;

            Eigen::Matrix4f guess;
            guess<< Rprev(0,0), Rprev(0,1), Rprev(0,2), tprev(0),
                    Rprev(1,0), Rprev(1,1), Rprev(1,2), tprev(1),
                    Rprev(2,0), Rprev(2,1), Rprev(2,2), tprev(2),
                    0,          0,          0,          1       ;
            icp.align(Final, guess);
            std::cout << "has converged?:" << icp.hasConverged() << " score: " <<
                         icp.getFitnessScore() << std::endl;
            std::cout << icp.getFinalTransformation() << std::endl;

            Eigen::Matrix4f final_icp_xf = icp.getFinalTransformation ();

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> temp;
            temp(0,0) = final_icp_xf(0,0); temp(0,1) = final_icp_xf(0,1); temp(0,2) = final_icp_xf(0,2);
            temp(1,0) = final_icp_xf(1,0); temp(1,1) = final_icp_xf(1,1); temp(1,2) = final_icp_xf(1,2);
            temp(2,0) = final_icp_xf(2,0); temp(2,1) = final_icp_xf(2,1); temp(2,2) = final_icp_xf(2,2);

            if(icp.hasConverged())
            {
                conv++;
                Rcurr = temp;
                tcurr = Vector3f(final_icp_xf(0,3), final_icp_xf(1,3), final_icp_xf(2,3));

                rmats_[process_t] = Rcurr;
                tvecs_[process_t] = tcurr;
            }
            cout<<"cam guess: "<<endl<< guess(0,0)<<" "<< guess(0,1)<<" "<<guess(0,2)<<" "<<guess(0,3)<<endl
               << guess(1,0)<<" "<< guess(1,1)<<" "<<guess(1,2)<<" "<<guess(1,3)<<endl
               << guess(2,0)<<" "<< guess(2,1)<<" "<<guess(2,2)<<" "<<guess(2,3)<<endl;

            cout<<"cam pose: "<<endl<< Rcurr(0,0)<<" "<< Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<tcurr(0)<<endl
               << Rcurr(1,0)<<" "<< Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<tcurr(1)<<endl
               << Rcurr(2,0)<<" "<< Rcurr(2,1)<<" "<<Rcurr(2,2)<<" "<<tcurr(2)<<endl<<endl;

            all_cloud = all_cloud.operator +=(Final);
        }

//        pcl::transformPointCloudWithNormals (src, src, xf);
//        all_cloud = all_cloud.operator +=(*src);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Volume integration
        float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

        Matrix3frm Rcurr_inv = Rcurr.inverse ();
        Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
        float3& device_tcurr = device_cast<float3> (tcurr);

        {
            stable_Rcam_ = Rcurr;
            stable_tcam_ = tcurr;
            cout<<" i"<<process_t<<" \n";
            integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_cast<Mat33> (Rcurr), device_tcurr, tsdf_volume_->getTsdfTruncDist(),
                               tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_isset, nmaps_curr_[0]);
        }

    }
//    pcl::io::savePLYFile ("outputs/ICP_cloud.ply", all_cloud);

    cout<<"Number of converged frames: "<<conv<<" out of "<<global_time_<<endl;
    return (true);
//    if(global_time_ < 1)
//        return (false);

//    if(cloud->points.size() < 100 )
//        return false;



//    cout<<"global time: "<<global_time_<<endl;
//    cout<<"rmats_: "<<rmats_.size()<<endl;

//    //clear voxel representation
//    tsdf_volume_->reset();

//    device::Intr intr (fx_, fy_, cx_, cy_);


//    bool roi_isset = false;


//    std::vector<int> indices;
//    cloud->is_dense = false;
//    cout<<cloud->points.size()<<endl;
//    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
//    cout<<cloud->points.size()<<" "<< indices.size() <<endl<<endl;

//    pcl::PointCloud<pcl::PointNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointNormal>);
//    tgt->is_dense = false;
//    pcl::copyPointCloud(*cloud, *tgt);



//    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
//    norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
//    //Set the number of k nearest neighbors to use for the feature estimation.
//    norm_est.setKSearch (40);
//    norm_est.setInputCloud (tgt);
//    norm_est.compute (*tgt);

//    //set point-to-plane distance metric
//    pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp_nl;
////    pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> icp;
//    typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
//    boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
//    icp_nl.setTransformationEstimation(point_to_plane);
//    icp_nl.setInputTarget(tgt);

////    icp_nl.setRANSACOutlierRejectionThreshold(0.1);
////    icp_nl.setRANSACIterations(100);
//    icp_nl.setTransformationEpsilon(1e-6);

//    icp_nl.setMaxCorrespondenceDistance (0.02); // in meters
//    icp_nl.setMaximumIterations(100);


//     pcl::PointCloud<pcl::PointNormal> all_cloud;// = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

//     int conv=0;
//    for(int process_t = 0; process_t < global_time_; process_t++)
//    {
//        if(process_t == roi_selection_time)
//            roi_isset = true;

//        cout<<process_t<<endl;
//        DepthMap depth_raw; //raw depth image
//        depth_raw.upload(depth_images.at(process_t), cols());

//        //get current frame vmap, nmap
//        {
//            device::bilateralFilter (depth_raw, depths_curr_[0]);

//            if (max_icp_distance_ > 0)
//                device::truncateDepth(depths_curr_[0], max_icp_distance_);

//            for (int i = 1; i < LEVELS; ++i)
//              device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

//            for (int i = 0; i < LEVELS; ++i)
//            {
//              device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
//              computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
//            }
//            pcl::device::sync ();
//        }

//        //////////////////////////////////////////////////
//        // ICP
//        /////////////////////////////////////////////////
//        Matrix3frm Rprev = rmats_[process_t]; //  [Ri|ti] - pos of camera, i.e.
//        Vector3f   tprev = tvecs_[process_t]; //  tranfrom from camera to global coo space for (i)th camera pose

//        Matrix3frm Rcurr = Rprev;
//        Vector3f   tcurr = tprev;

//        DeviceArray2D<PointXYZ> cloud_device_;
//        cloud_device_.create (rows_, cols_);
//        DeviceArray2D<float4>& cd = (DeviceArray2D<float4>&)cloud_device_;
//        device::convert (vmaps_curr_[0], cd);

//        int c;
//        cloud_device_.download (cloud_curr->points, c);
//        cloud_curr->width = cloud_device_.cols ();
//        cloud_curr->height = cloud_device_.rows ();
//        cloud_curr->is_dense = false;

//        pcl::PointCloud<pcl::PointNormal>::Ptr src(new pcl::PointCloud<pcl::PointNormal>);
//        pcl::copyPointCloud(*cloud_curr, *src);
//        src->width = cloud_device_.cols ();
//        src->height = cloud_device_.rows ();
//        src->is_dense = false;

//        //remove NAN points from the cloud
//        pcl::removeNaNFromPointCloud(*src,*src, indices);

////        char fn[50];
////        sprintf(fn, "outputs/frame_%4d.ply", process_t);
////        pcl::io::savePLYFile (fn, *src);

//        if(process_t==0) {
////            icp_nl.setInputTarget(src);
////            all_cloud = all_cloud.operator +=(*src);
//        }


//        if(process_t>0 && cloud_curr->points.size()>0)
//        {
//////            pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
//////            norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
//////            //Set the number of k nearest neighbors to use for the feature estimation.
//////            norm_est.setKSearch (40);
////            pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud_ptr = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
////            pcl::copyPointCloud(all_cloud, *all_cloud_ptr);
////            all_cloud_ptr->width = all_cloud.width;
////            all_cloud_ptr->height = all_cloud.height;
////            all_cloud_ptr->is_dense = false;
////            //remove NAN points from the cloud
////            pcl::removeNaNFromPointCloud(*all_cloud_ptr,*all_cloud_ptr, indices);

//////            norm_est.setInputCloud (all_cloud_ptr);
//////            norm_est.compute (all_cloud);

//////            typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
//////            boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
//////            icp_nl.setTransformationEstimation(point_to_plane);
//////            icp_nl.setInputTarget(all_cloud_ptr);


////            icp_nl.setInputTarget(all_cloud_ptr);

//            icp_nl.setInputCloud(src);

//            pcl::PointCloud<pcl::PointNormal> Final;


//            Matrix3frm Rinc ;//= rmats_icp_nl[process_t-1]; //  [Ri|ti] - pos of camera, i.e.
//            Vector3f   tinc ;//= tvecs_icp_nl[process_t-1];

//            Rinc = Rcurr;
////            Rinc = Rcurr*Rinc.inverse();
////            tinc = tcurr - Rinc * tinc;

//            Eigen::Matrix4f guess;
//            guess<< Rinc(0,0), Rinc(0,1), Rinc(0,2), tinc(0),
//                    Rinc(1,0), Rinc(1,1), Rinc(1,2), tinc(1),
//                    Rinc(2,0), Rinc(2,1), Rinc(2,2), tinc(2),
//                    0,          0,          0,          1       ;

//            icp_nl.align(Final);


//            std::cout << "has converged?:" << icp_nl.hasConverged() << " score: " <<
//                         icp_nl.getFitnessScore() << std::endl;
//            std::cout << icp_nl.getFinalTransformation() << std::endl;

//            Eigen::Matrix4f final_icp_xf = icp_nl.getFinalTransformation ();

//            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> temp;
//            temp(0,0) = final_icp_xf(0,0); temp(0,1) = final_icp_xf(0,1); temp(0,2) = final_icp_xf(0,2);
//            temp(1,0) = final_icp_xf(1,0); temp(1,1) = final_icp_xf(1,1); temp(1,2) = final_icp_xf(1,2);
//            temp(2,0) = final_icp_xf(2,0); temp(2,1) = final_icp_xf(2,1); temp(2,2) = final_icp_xf(2,2);

//            if(icp_nl.hasConverged())
//            {
//                conv++;
//                tinc = Vector3f(final_icp_xf(0,3), final_icp_xf(1,3), final_icp_xf(2,3));

////                tcurr = temp*tvecs_icp_nl[process_t] + tinc;
////                Rcurr = temp*rmats_icp_nl[process_t];

//                Rcurr = temp;
//                tcurr = tinc;

//                tvecs_[process_t] = tcurr ;
//                rmats_[process_t] = Rcurr;

//                cout<<"cam guess: "<<endl<< guess(0,0)<<" "<< guess(0,1)<<" "<<guess(0,2)<<" "<<guess(0,3)<<endl
//                   << guess(1,0)<<" "<< guess(1,1)<<" "<<guess(1,2)<<" "<<guess(1,3)<<endl
//                   << guess(2,0)<<" "<< guess(2,1)<<" "<<guess(2,2)<<" "<<guess(2,3)<<endl;

//                cout<<"cam pose: "<<endl<< Rcurr(0,0)<<" "<< Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<tcurr(0)<<endl
//                   << Rcurr(1,0)<<" "<< Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<tcurr(1)<<endl
//                   << Rcurr(2,0)<<" "<< Rcurr(2,1)<<" "<<Rcurr(2,2)<<" "<<tcurr(2)<<endl<<endl;
//            }


//            all_cloud = all_cloud.operator +=(Final);
////            icp_nl.setInputTarget(src);
//        }


//        ///////////////////////////////////////////////////////////////////////////////////////////
//        // Volume integration
//        float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

//        Matrix3frm Rcurr_inv = Rcurr.inverse ();
//        Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
//        float3& device_tcurr = device_cast<float3> (tcurr);

//        {
//            stable_Rcam_ = Rcurr;
//            stable_tcam_ = tcurr;
//            cout<<" i"<<process_t<<" \n";
//            integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_cast<Mat33> (Rcurr), device_tcurr, tsdf_volume_->getTsdfTruncDist(),
//                               tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_isset, nmaps_curr_[0]);
//        }

//    }
//    pcl::io::savePLYFile ("outputs/ICP_cloud.ply", all_cloud);

//    cout<<"Number of converged frames: "<<conv<<" out of "<<global_time_<<endl;
//    return (true);
};

bool
pcl::gpu::KinfuTracker::reconstructWithModelProxy (PointCloud<PointXYZ>::Ptr cloud, int ctrl)
{
    if(global_time_ < 1)
        return (false);

    if(cloud->points.size() < 100 )
        return false;


    ofstream path_file_stream("outputs/fitness_scores_icp_plane.txt");
//    path_file_stream.setf(ios::fixed,ios::doubleField);

    cout<<"global time: "<<global_time_<<endl;
    cout<<"rmats_: "<<rmats_.size()<<endl;


    //clear voxel representation
    tsdf_volume_->reset();

    device::Intr intr (fx_, fy_, cx_, cy_);


    bool roi_isset = false;


    std::vector<int> indices;
    cloud->is_dense = false;
    cout<<cloud->points.size()<<endl;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    cout<<cloud->points.size()<<" "<< indices.size() <<endl<<endl;

    pcl::PointCloud<pcl::PointNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointNormal>);
    tgt->is_dense = false;
    pcl::copyPointCloud(*cloud, *tgt);



    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
    norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
    //Set the number of k nearest neighbors to use for the feature estimation.
    norm_est.setKSearch (40);
    norm_est.setInputCloud (tgt);
    norm_est.compute (*tgt);

    //set point-to-plane distance metric
    pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
    typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
    if(ctrl == 1) {
        boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
        icp.setTransformationEstimation(point_to_plane);
    }
    icp.setInputTarget(tgt);

    //      icp.setRANSACOutlierRejectionThreshold(0.1);
//    icp.setRANSACIterations(100);
    icp.setTransformationEpsilon(1e-6);

    icp.setMaxCorrespondenceDistance (0.02); // in meters
    icp.setMaximumIterations(100);


     pcl::PointCloud<pcl::PointNormal> all_cloud;// = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);;

     int conv=0;
    for(int process_t = 0; process_t < global_time_; process_t++)
    {
        path_file_stream<<process_t<<"  ";
        if(process_t == roi_selection_time)
            roi_isset = true;

        cout<<process_t<<endl;
        DepthMap depth_; // depth image
        depth_.upload(depth_images.at(process_t), cols());

        //get current frame vmap, nmap
        {
//            device::bilateralFilter (depth_raw, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(), scale_factor);

             depths_curr_[0] = depth_;

            if (max_icp_distance_ > 0)
                device::truncateDepth(depths_curr_[0], max_icp_distance_, scale_factor);

            for (int i = 1; i < LEVELS; ++i)
              device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

            for (int i = 0; i < LEVELS; ++i)
            {
              device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
              computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
            }
            pcl::device::sync ();
        }

        //////////////////////////////////////////////////
        // ICP
        /////////////////////////////////////////////////
        Matrix3frm Rprev = rmats_[process_t]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   tprev = tvecs_[process_t]; //  tranfrom from camera to global coo space for (i)th camera pose

        Matrix3frm Rcurr = Rprev;
        Vector3f   tcurr = tprev;

        DeviceArray2D<PointXYZ> cloud_device_;
        cloud_device_.create (rows_, cols_);
        DeviceArray2D<float4>& cd = (DeviceArray2D<float4>&)cloud_device_;
        device::convert (vmaps_curr_[0], cd);

        int c;
        cloud_device_.download (cloud_curr->points, c);
        cloud_curr->width = cloud_device_.cols ();
        cloud_curr->height = cloud_device_.rows ();
        cloud_curr->is_dense = false;


        if(cloud_curr->points.size()>0)
        {

            pcl::PointCloud<pcl::PointNormal>::Ptr src(new pcl::PointCloud<pcl::PointNormal>);
            pcl::copyPointCloud(*cloud_curr, *src);
            src->width = cloud_device_.cols ();
            src->height = cloud_device_.rows ();
            src->is_dense = false;


            cout<<src->points.size()<<endl;
            //remove NAN points from the cloud
            pcl::removeNaNFromPointCloud(*src,*src, indices);



            icp.setInputCloud(src);

            cout<<src->points.size()<<endl;
            cout<<tgt->points.size()<<endl;

            pcl::PointCloud<pcl::PointNormal> Final;

            Eigen::Matrix4f guess;
            guess<< Rprev(0,0), Rprev(0,1), Rprev(0,2), tprev(0),
                    Rprev(1,0), Rprev(1,1), Rprev(1,2), tprev(1),
                    Rprev(2,0), Rprev(2,1), Rprev(2,2), tprev(2),
                    0,          0,          0,          1       ;
            icp.align(Final, guess);
            std::cout << "has converged?:" << icp.hasConverged() << " score: " <<
                         icp.getFitnessScore() << std::endl;
            std::cout << icp.getFinalTransformation() << std::endl;

            path_file_stream<<icp.getFitnessScore();
            Eigen::Matrix4f final_icp_xf = icp.getFinalTransformation ();

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> temp;
            temp(0,0) = final_icp_xf(0,0); temp(0,1) = final_icp_xf(0,1); temp(0,2) = final_icp_xf(0,2);
            temp(1,0) = final_icp_xf(1,0); temp(1,1) = final_icp_xf(1,1); temp(1,2) = final_icp_xf(1,2);
            temp(2,0) = final_icp_xf(2,0); temp(2,1) = final_icp_xf(2,1); temp(2,2) = final_icp_xf(2,2);

            if(icp.hasConverged())
            {
                conv++;
                Rcurr = temp;
                tcurr = Vector3f(final_icp_xf(0,3), final_icp_xf(1,3), final_icp_xf(2,3));

                rmats_[process_t] = Rcurr;
                tvecs_[process_t] = tcurr;
            }
            cout<<"cam guess: "<<endl<< guess(0,0)<<" "<< guess(0,1)<<" "<<guess(0,2)<<" "<<guess(0,3)<<endl
               << guess(1,0)<<" "<< guess(1,1)<<" "<<guess(1,2)<<" "<<guess(1,3)<<endl
               << guess(2,0)<<" "<< guess(2,1)<<" "<<guess(2,2)<<" "<<guess(2,3)<<endl;

            cout<<"cam pose: "<<endl<< Rcurr(0,0)<<" "<< Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<tcurr(0)<<endl
               << Rcurr(1,0)<<" "<< Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<tcurr(1)<<endl
               << Rcurr(2,0)<<" "<< Rcurr(2,1)<<" "<<Rcurr(2,2)<<" "<<tcurr(2)<<endl<<endl;

            all_cloud = all_cloud.operator +=(Final);
            //write the current cloud into a .3d file
            char fn[50];
            sprintf(fn, "outputs/pt_clouds/scan%03d.3d", process_t);
            ofstream scan_file_3d(fn);

            for(size_t i=0; i<Final.size(); i++)
            {
                pcl::PointNormal p = Final.at(i);
                scan_file_3d<<10.*p.x<<"    "<<10.*p.y<<"   "<<10.*p.z<<endl;
            }
            scan_file_3d.close();
            sprintf(fn, "outputs/pt_clouds/scan%03d.pose", process_t);
            ofstream scan_file_pose(fn);

            scan_file_pose<<0<<"    "<<0<<"    "<<0<<endl
                            <<0<<"    "<<0<<"    "<<0;

            scan_file_pose.close();

        }

        path_file_stream<<endl;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Volume integration
        float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());


        DepthMap depth_raw; // depth image
        depth_raw.upload(depth_raw_images.at(process_t), cols());

        device::scaleRawDepth(depth_raw, depth_raw, 64.0, 1./64.);

        Matrix3frm Rcurr_inv = Rcurr.inverse ();
        Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
        float3& device_tcurr = device_cast<float3> (tcurr);

        {
            stable_Rcam_ = Rcurr;
            stable_tcam_ = tcurr;
            cout<<" i"<<process_t<<" \n";
            integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_cast<Mat33> (Rcurr), device_tcurr, tsdf_volume_->getTsdfTruncDist(),
                               tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_isset, nmaps_curr_[0]);
        }

    }
    pcl::io::savePLYFile ("outputs/ICP_cloud.ply", all_cloud);

    cout<<"Number of converged frames: "<<conv<<" out of "<<global_time_<<endl;
    return (true);
};

void
pcl::gpu::KinfuTracker::raw_depth_to_bilateraled()
{
    this->setScaleFactor(64.);
    for(int process_t = 0; process_t < global_time_; process_t++)
    {

        DepthMap depth_raw(rows(), cols()); //raw depth image
        depth_raw.upload(depth_images.at(process_t), cols());

        {
            device::bilateralFilter (depth_raw, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(), 64.);

            if (max_icp_distance_ > 0)
                device::truncateDepth(depths_curr_[0], max_icp_distance_, 64.);

            pcl::device::sync ();
        }
        int colss;
        vector<ushort> depth_host_;
        depths_curr_[0].download (depth_host_, colss);
        depth_images[process_t] = depth_host_;
    }
}

bool
pcl::gpu::KinfuTracker::reconstructWithModelProxy2 (PointCloud<PointXYZ>::Ptr cloud, int ctrl)
{
    if(global_time_ < 1)
        return (false);

    if(cloud->points.size() < 100 )
        return false;


    ofstream path_file_stream("outputs/fitness_scores_test.txt");
//    path_file_stream.setf(ios::fixed,ios::floatfield);

    cout<<"global time: "<<global_time_<<endl;
    cout<<"rmats_: "<<rmats_.size()<<endl;

    //clear voxel representation
    tsdf_volume_->reset();

    device::Intr intr (fx_, fy_, cx_, cy_);


    bool roi_isset = false;


    std::vector<int> indices;
    cloud->is_dense = false;
    cout<<cloud->points.size()<<endl;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    cout<<cloud->points.size()<<" "<< indices.size() <<endl<<endl;

//    pcl::PointCloud<pcl::PointNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointNormal>);
//    tgt->is_dense = false;
//    pcl::copyPointCloud(*cloud, *tgt);



//    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
//    norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
//    //Set the number of k nearest neighbors to use for the feature estimation.
//    norm_est.setKSearch (40);
//    norm_est.setInputCloud (tgt);
//    norm_est.compute (*tgt);

    //set point-to-plane distance metric
    pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
    typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
    if(ctrl == 1) {
        boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
        icp.setTransformationEstimation(point_to_plane);
    }
//    icp.setInputTarget(tgt);

    //      icp.setRANSACOutlierRejectionThreshold(0.1);
//    icp.setRANSACIterations(100);
    icp.setTransformationEpsilon(1e-6);

    icp.setMaxCorrespondenceDistance (0.02); // in meters
    icp.setMaximumIterations(100);


     pcl::PointCloud<pcl::PointNormal> all_cloud;// = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);;

     int conv=0;
    for(int process_t = 0; process_t < global_time_; process_t++)
    {
        path_file_stream<<process_t;
        if(process_t == roi_selection_time)
            roi_isset = true;

        cout<<process_t<<endl;
        DepthMap depth_raw; //raw depth image
        depth_raw.upload(depth_images.at(process_t), cols());

        //get current frame vmap, nmap
        {
            device::bilateralFilter (depth_raw, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(), scale_factor);

            if (max_icp_distance_ > 0)
                device::truncateDepth(depths_curr_[0], max_icp_distance_, scale_factor);

            for (int i = 1; i < LEVELS; ++i)
              device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

            for (int i = 0; i < LEVELS; ++i)
            {
              device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
              computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
            }
            pcl::device::sync ();
        }

        //////////////////////////////////////////////////
        // ICP
        /////////////////////////////////////////////////
        Matrix3frm Rprev = rmats_[process_t]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   tprev = tvecs_[process_t]; //  tranfrom from camera to global coo space for (i)th camera pose

        Matrix3frm Rcurr = Rprev;
        Vector3f   tcurr = tprev;

        DeviceArray2D<PointXYZ> cloud_device_;
        cloud_device_.create (rows_, cols_);
        DeviceArray2D<float4>& cd = (DeviceArray2D<float4>&)cloud_device_;
        device::convert (vmaps_curr_[0], cd);

        int c;
        cloud_device_.download (cloud_curr->points, c);
        cloud_curr->width = cloud_device_.cols ();
        cloud_curr->height = cloud_device_.rows ();
        cloud_curr->is_dense = false;

        pcl::PointCloud<pcl::PointNormal>::Ptr src(new pcl::PointCloud<pcl::PointNormal>);
        pcl::copyPointCloud(*cloud_curr, *src);
        src->width = cloud_device_.cols ();
        src->height = cloud_device_.rows ();
        src->is_dense = false;

        //remove NAN points from the cloud
        pcl::removeNaNFromPointCloud(*src,*src, indices);

        if(process_t == 0)
        {
            Eigen::Matrix4f xf;
            xf<<    Rprev(0,0), Rprev(0,1), Rprev(0,2), tprev(0),
                    Rprev(1,0), Rprev(1,1), Rprev(1,2), tprev(1),
                    Rprev(2,0), Rprev(2,1), Rprev(2,2), tprev(2),
                    0,          0,          0,          1       ;
            pcl::transformPointCloudWithNormals (*src, *src, xf);
            all_cloud = all_cloud.operator +=(*src);
        }
        else if(cloud_curr->points.size()>0)
        {
            pcl::PointCloud<pcl::PointNormal>::Ptr all_cloud_ptr = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
            pcl::copyPointCloud(all_cloud, *all_cloud_ptr);
            all_cloud_ptr->width = all_cloud.width;
            all_cloud_ptr->height = all_cloud.height;
            all_cloud_ptr->is_dense = false;
            //remove NAN points from the cloud
            pcl::removeNaNFromPointCloud(*all_cloud_ptr,*all_cloud_ptr, indices);

            pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
            norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointNormal>::Ptr (new pcl::search::KdTree<pcl::PointNormal>));
            //Set the number of k nearest neighbors to use for the feature estimation.
            norm_est.setKSearch (40);
            norm_est.setInputCloud (all_cloud_ptr);
            norm_est.compute (*all_cloud_ptr);

            typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
            boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
            icp.setTransformationEstimation(point_to_plane);

            icp.setInputTarget(all_cloud_ptr);


            icp.setInputCloud(src);

            cout<<src->points.size()<<endl;
//            cout<<tgt->points.size()<<endl;

            pcl::PointCloud<pcl::PointNormal> Final;

            Eigen::Matrix4f guess;
            guess<< Rprev(0,0), Rprev(0,1), Rprev(0,2), tprev(0),
                    Rprev(1,0), Rprev(1,1), Rprev(1,2), tprev(1),
                    Rprev(2,0), Rprev(2,1), Rprev(2,2), tprev(2),
                    0,          0,          0,          1       ;
            icp.align(Final, guess);
            std::cout << "has converged?:" << icp.hasConverged() << " score: " <<
                         icp.getFitnessScore() << std::endl;
            path_file_stream<<" "<<icp.getFitnessScore();
            std::cout << icp.getFinalTransformation() << std::endl;

            Eigen::Matrix4f final_icp_xf = icp.getFinalTransformation ();

            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> temp;
            temp(0,0) = final_icp_xf(0,0); temp(0,1) = final_icp_xf(0,1); temp(0,2) = final_icp_xf(0,2);
            temp(1,0) = final_icp_xf(1,0); temp(1,1) = final_icp_xf(1,1); temp(1,2) = final_icp_xf(1,2);
            temp(2,0) = final_icp_xf(2,0); temp(2,1) = final_icp_xf(2,1); temp(2,2) = final_icp_xf(2,2);

            if(icp.hasConverged())
            {
                conv++;
                Rcurr = temp;
                tcurr = Vector3f(final_icp_xf(0,3), final_icp_xf(1,3), final_icp_xf(2,3));

                rmats_[process_t] = Rcurr;
                tvecs_[process_t] = tcurr;
            }
            cout<<"cam guess: "<<endl<< guess(0,0)<<" "<< guess(0,1)<<" "<<guess(0,2)<<" "<<guess(0,3)<<endl
               << guess(1,0)<<" "<< guess(1,1)<<" "<<guess(1,2)<<" "<<guess(1,3)<<endl
               << guess(2,0)<<" "<< guess(2,1)<<" "<<guess(2,2)<<" "<<guess(2,3)<<endl;

            cout<<"cam pose: "<<endl<< Rcurr(0,0)<<" "<< Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<tcurr(0)<<endl
               << Rcurr(1,0)<<" "<< Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<tcurr(1)<<endl
               << Rcurr(2,0)<<" "<< Rcurr(2,1)<<" "<<Rcurr(2,2)<<" "<<tcurr(2)<<endl<<endl;

            all_cloud = all_cloud.operator +=(Final);
        }

        path_file_stream<<endl;
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Volume integration
        float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

        Matrix3frm Rcurr_inv = Rcurr.inverse ();
        Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
        float3& device_tcurr = device_cast<float3> (tcurr);

        {
            stable_Rcam_ = Rcurr;
            stable_tcam_ = tcurr;
            cout<<" i"<<process_t<<" \n";
            integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_cast<Mat33> (Rcurr), device_tcurr, tsdf_volume_->getTsdfTruncDist(),
                               tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_isset, nmaps_curr_[0]);
        }

    }
    pcl::io::savePLYFile ("outputs/ICP_cloud.ply", all_cloud);

    cout<<"Number of converged frames: "<<conv<<" out of "<<global_time_<<endl;
    return (true);
};


//    int colsss;
//    vector<ushort> depth_host_1;
//    depths_curr_[0].download (depth_host_1, colsss);

//    cv::Mat frame;
//    frame.create (cv::Size (640, 480), CV_16U);

//    for(int x=0; x<frame.rows; x++)
//        for(int y=0; y<frame.cols; y++)
//        {
//            int c = depth_host_1[x*colsss+y];

//            frame.at<short>(x, y) = c;
//        }

//    std::vector<int> compression_params;
//        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//        compression_params.push_back(0);

//    char fn[50];
//    sprintf(fn, "depth_%d.png", (int)global_time_);
//    try {
//        cv::imwrite(fn, frame, compression_params);
//    }
//    catch (std::runtime_error& ex) {
//        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        return 1;
//    }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth_raw, View& view, bool pause, bool roi_selected_, View& rgb24)
{

//    DepthMap depth_(depth_raw1);
//    device::removeNonobjectPoints(depth_raw, view);
//    //////////////////////////
//    //SEMA
//    if(pause)
//    {
//        device::removeNonobjectPoints(depth_raw, view);
//    }

    if(ishand)
    {
        DepthMap depth_(depth_raw);
        device::maskRGB (depth_, rgb24, max_icp_distance_);


        int col;
        vector<KinfuTracker::PixelRGB> rgb_host_;
        rgb24.download (rgb_host_, col);

        for(int x=0; x<2*rows(); x++) {
            for(int y=0; y<2*cols(); y++)
            {
                PixelRGB c = rgb_host_[x*col+y];

                if((int)c.r != 0 && (int)c.g != 0 && (int)c.b != 0)
                    hand.samples.push_back(c);
            }
        }
        return 0;
    }
    isICPfail = false;

    DepthMap depth_raw1(depth_raw);
    DepthMap depth_blur(rows(), cols());
  {
//      device::bilateralFilter (depth_raw, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(), scale_factor);

//      device::blurFilter (depths_curr_[0], depth_blur);

//      depth_raw.copyTo(depth_raw1);

      if(hand.isset) {
          device::scaleRawDepth(depth_raw, depth_blur,  max_icp_distance_, 1);
          depth_blur.copyTo(depth_raw1);

          int col;
          vector<KinfuTracker::PixelRGB> rgb_host_;
          rgb24.download (rgb_host_, col);

          int colss;
//          vector<ushort> depth_host;
//          depth_blur.download (depth_host, colss);

          vector<ushort> depth_host;
          depth_blur.download (depth_host, colss);

          float s = hand.scale;

          for(int x=0; x<rows(); x++){
              for(int y=0; y<cols(); y++)
              {
//                  if(depth_host[x*cols()+y] < 500*64) {
//                      depth_host[x*cols()+y] = 0;
//                      depth_raw_host[x*cols()+y] = 0;
//                      continue;
//                  }

                  PixelRGB c = rgb_host_[2*x*col+2*y];

                  int i = round(((int)c.r)*s);
                  int j = round(((int)c.g)*s);
                  int k = round(((int)c.b)*s);

                  i = std::max(0, std::min(i, hand.cube_dim-1));
                  j = std::max(0, std::min(j, hand.cube_dim-1));
                  k = std::max(0, std::min(k, hand.cube_dim-1));


                  if(hand.color_cube[i][j][k] > 0){
//                      depth_host[x*cols()+y] = 0;
                      depth_host[x*cols()+y] = 0;
                  }
              }
          }
//          depth_blur.upload(depth_host, cols());
          depth_blur.upload(depth_host, cols());
          device::erode(depth_blur, depth_raw1);
      }

//      depth_blur.copyTo(depth_raw1);
      pcl::device::sync ();
  }

  device::Intr intr (fx_, fy_, cx_, cy_);
  {
    //ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
//device::scaleRawDepth(depth_raw, depth_raw1,  max_icp_distance_, 1);
    device::bilateralFilter (depth_raw1, depths_curr_[0], tsdf_volume_->getTsdfTruncDist(), scale_factor);

    int colss;
    vector<ushort> depth_host_;
    DepthMap depth_raw_save(rows(), cols());
    device::scaleRawDepth(depth_raw1, depth_raw_save,  max_icp_distance_, 64);
    depth_raw_save.download (depth_host_, colss);
    depth_raw_images.push_back(depth_host_);
    depth_host_.clear();
    depths_curr_[0].download (depth_host_, colss);
    depth_images.push_back(depth_host_);

    if (max_icp_distance_ > 0)
        device::truncateDepth(depths_curr_[0], max_icp_distance_, scale_factor);



    for (int i = 1; i < LEVELS; ++i)
      device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

    for (int i = 0; i < LEVELS; ++i)
    {
      device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
      //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
      computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);

//      if(i==0)
//        device::eliminateVMapIfNotPerpendicular (vmaps_curr_[i], nmaps_curr_[i], depths_curr_[0], depth_);
//      else
//        device::eliminateVMapIfNotPerpendicular (vmaps_curr_[i], nmaps_curr_[i]);

    }

//    device::blurFilter (depths_curr_[0], depth_blur);
    device::maskRGB (depth_raw1, rgb24, max_icp_distance_*scale_factor);

    //sema
//    device::createVMap (intr(0), depth_raw1, vmap_save);
//    computeNormalsEigen (vmap_save, nmap_save);
//    vmaps_.push_back(vmap_save);
//    nmaps_.push_back(nmap_save);

    //SEMA - remove ground plane for each frame
//    getBasePlane();
//    device::eliminatePointsOnGround(depth_raw, vmaps_curr_[0], *plane_coeffs);

    pcl::device::sync ();
  }

  //can't perform more on first frame
  if (global_time_ == 0)
  {
    Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
    Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

    Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
    float3& device_tcam = device_cast<float3>(init_tcam);

    Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
    Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
    float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

    r_inv_.push_back(device_Rcam_inv);
    t_.push_back(device_tcam);
    //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);
    device::integrateTsdfVolume(depth_raw1, intr, device_volume_size, device_Rcam_inv, device_Rcam, device_tcam, tsdf_volume_->getTsdfTruncDist(),
                                tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_selected_, nmaps_curr_[0]);

    for (int i = 0; i < LEVELS; ++i)
      device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);

    ++global_time_;

    return (false);
  }

//  //SEMA
//  //image registration
//  rgb_prev.operator =(rgb_curr);
//  rgb_curr.operator =(rgb_mat);
//  calculateImageRegistrationSIFT();




  ///////////////////////////////////////////////////////////////////////////////////////////
  // Iterative Closest Point
  Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
  Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
  Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

  //Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
  Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
  float3& device_tprev     = device_cast<float3> (tprev);

//  //sema
//  //linear model for initial motion estimation
//  Matrix3frm Rprevprev = Rprev;
//  Vector3f   tprevprev = tprev;
//  if(global_time_>1) {
//      Rprevprev = rmats_[global_time_ - 2];
//      tprevprev = tvecs_[global_time_ - 2];
//  }
//  Matrix3frm Rcurr = ( Rprev*( Rprevprev.inverse() ) ) * Rprev; // tranform to global coo for ith camera pose
//  Vector3f   tcurr = 2*tprev-tprevprev;

    Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
    Vector3f   tcurr = tprev;

  {
    //ScopeTime time("icp-all");
    //coarse to fine icp
    for (int level_index = LEVELS-1; level_index>=0; --level_index)
    {
      int iter_num = icp_iterations_[level_index]; // [10,5,4]

      MapArr& vmap_curr = vmaps_curr_[level_index];
      MapArr& nmap_curr = nmaps_curr_[level_index];

      //MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
      //MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

      MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
      MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

      //CorespMap& coresp = coresps_[level_index];

      for (int iter = 0; iter < iter_num; ++iter)
      {
        Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
        float3& device_tcurr = device_cast<float3>(tcurr);

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
        Eigen::Matrix<double, 6, 1> b;
#if 0
        device::tranformMaps(vmap_curr, nmap_curr, device_Rcurr, device_tcurr, vmap_g_curr, nmap_g_curr);
        findCoresp(vmap_g_curr, nmap_g_curr, device_Rprev_inv, device_tprev, intr(level_index), vmap_g_prev, nmap_g_prev, distThres_, angleThres_, coresp);
        device::estimateTransform(vmap_g_prev, nmap_g_prev, vmap_g_curr, coresp, gbuf_, sumbuf_, A.data(), b.data());

        //cv::gpu::GpuMat ma(coresp.rows(), coresp.cols(), CV_32S, coresp.ptr(), coresp.step());
        //cv::Mat cpu;
        //ma.download(cpu);
        //cv::imshow(names[level_index] + string(" --- coresp white == -1"), cpu == -1);
#else
        estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
                          vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
#endif
        //checking nullspace
        double det = A.determinant ();

         Eigen::Matrix<double, 6, 1> result = result = A.llt ().solve (b).cast<double>();
        if(fabs (det) < 1e-15 || pcl_isnan (det))
        {
//            det = ( A.transpose()*A ).determinant ();
//            if(fabs (det) < 1e-15 || pcl_isnan (det))
//            {
                if (pcl_isnan (det)){ reset ();   cout << "qnan" << endl;}

                depth_images.pop_back();
                depth_raw_images.pop_back();

//                cout<<"failure at icp level: "<<level_index<<endl;

                isICPfail = true;

                return (false);

//            }
//            result =  ( A.transpose()*A ).inverse() * A.transpose() * b;
        }
//        Eigen::MatrixXd tt = Eigen::MatrixXd::Random(6,6);
//        Eigen::Matrix<double, 6, 1> result = tt.llt ().solve (b).cast<double>();

//        if (fabs (det) < 1e-15 || pcl_isnan (det))
//        {
//            if (pcl_isnan (det)){ reset ();   cout << "qnan" << endl;}

//            depth_images.pop_back();
//            depth_raw_images.pop_back();

//            cout<<"failure at icp level: "<<level_index<<endl;

//            return (false);

//        }
        //float maxc = A.maxCoeff();

//        Eigen::Matrix<double, 6, 1> result = ( A.transpose()*A ).inverse() * A.transpose() * b;

//        Eigen::Matrix<float, 6, 1> result = static_cast< Eigen::Matrix<float, 6, 1> >(x);
//        Eigen::Matrix<float, 6, 1> result;// = A*b;//(A.transpose()*A).inverse()*A.transpose()*b;


//        Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();


//        Eigen::Matrix<double, 6, 1> result = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        float alpha = result (0);
        float beta  = result (1);
        float gamma = result (2);
        Vector3f tinc;// = result.tail<3> ();
        tinc[0] = result (3);
        tinc[1] = result (4);
        tinc[2] = result (5);
//        if(alpha == beta == gamma == 0 && tinc == Vector3f(0,0,0))
//        {
//            continue;
////            if (pcl_isnan (det)){ reset ();   cout << "qnan" << endl;}

//            depth_images.pop_back();
//            depth_raw_images.pop_back();

//            cout<<"failure at icp level: "<<level_index<<endl;

//            isICPfail = true;
//            return (false);

//        }

        Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());


//        cout<<alpha<<" "<<beta<<" "<<gamma<<endl;
//        cout<<tinc<<endl;

        //compose
        tcurr = Rinc * tcurr + tinc;
        Rcurr = Rinc * Rcurr;
      }
    }
  }

    float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
    float tnorm = (tcurr - tprev).norm();
    const float alpha = 1.f;
    float move_norm = (rnorm + alpha * tnorm)/2;

//    if(move_norm > 1)
//    {
////            continue;
////            if (pcl_isnan (det)){ reset ();   cout << "qnan" << endl;}

//        depth_images.pop_back();
//        depth_raw_images.pop_back();

//        cout<<"ICP failed!!!!"<<endl;

//        isICPfail = true;
//        return (false);

//    }


  //save tranform
  rmats_.push_back (Rcurr);
  tvecs_.push_back (tcurr);

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move.
//  float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
//  float tnorm = (tcurr - tprev).norm();
//  const float alpha = 1.f;
//  float move_norm = (rnorm + alpha * tnorm)/2;
////  cout<<move_norm<<endl;
  bool integrate =  (move_norm >= 0);// && (move_norm < 10);//integration_metric_threshold_);


  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  Matrix3frm Rcurr_inv = Rcurr.inverse ();
  Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
  float3& device_tcurr = device_cast<float3> (tcurr);

//  r_inv_.push_back(device_Rcurr_inv);
//  t_.push_back(device_tcurr);

  if (integrate && !pause)
  {
      stable_Rcam_ = Rcurr;
      stable_tcam_ = tcurr;

    //ScopeTime time("tsdf");
    //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
    integrateTsdfVolume (depth_raw1, intr, device_volume_size, device_Rcurr_inv, device_cast<Mat33> (Rcurr), device_tcurr, tsdf_volume_->getTsdfTruncDist(),
                         tsdf_volume_->data(), depthRawScaled_, roi_boundaries_, roi_selected_, nmaps_curr_[0]);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
  {
    //ScopeTime time("ray-cast-all");

    raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0], view, rgb24);

    view.copyTo(raycast_view);

    //frame-to-frame icp
//    for (int i = 0; i < LEVELS; ++i)
//      device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcurr, device_tcurr, vmaps_g_prev_[i], nmaps_g_prev_[i]);

    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }

    pcl::device::sync ();
  }

  ++global_time_;

  return (true);
}


//sema
void fill_video_frame(cv::Mat& video_frame, KinfuTracker::View rgb24)
{
    int cols;
    vector<KinfuTracker::PixelRGB> rgb_host_;
    rgb24.download (rgb_host_, cols);

    for(int x=0; x<video_frame.rows; x++)
        for(int y=0; y<video_frame.cols; y++)
        {
            PixelRGB c = rgb_host_[x*cols+y];
            video_frame.at<cv::Vec3b>(x, y) = cv::Vec3b(c.b, c.g, c.r);  //BGR
        }
}

/////sema
void
pcl::gpu::KinfuTracker::calculateImageRegistrationSIFT()
{

    std::vector<cv::KeyPoint> key_prev, key_curr;
    cv::Mat desc_prev = cv::Mat(0,DESC_LENGTH, 5);
    cv::Mat desc_curr = cv::Mat(0,DESC_LENGTH, 5);


    opencvSIFT(rgb_curr, key_curr, desc_curr);

//    if(global_time_ == 1)
    opencvSIFT(rgb_prev, key_prev, desc_prev);

    cv::Ptr<cv::DescriptorMatcher>  desc_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector< std::vector<cv::DMatch> > matches;
    desc_matcher->radiusMatch(desc_prev, desc_curr, matches, 8.f);

//    std::vector<cv::DMatch> matches;
//    desc_matcher->match(desc_prev, desc_curr, matches);

//    cout<<"# matches: "<< matches.size()<<endl;


    //write into a file
    cv::Mat outImg;
    cv::drawMatches(rgb_prev, key_prev, rgb_curr, key_curr, matches, outImg);

    std::vector<int> compression_params;

    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    char fn[50];
    sprintf(fn, "video/mvs/keypoints_%08d.jpg", (int)global_time_);
    try {
        cv::imwrite(fn, outImg, compression_params);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
        return;
    }
}

void opencvSIFT(cv::Mat input_org, std::vector<cv::KeyPoint> &keypoints, cv::Mat& descriptors)
{
    unsigned t0=clock(),t1;
    cv::Mat input;

    if(input_org.type() == CV_8UC3)
        cvtColor( input_org, input, CV_BGR2GRAY );
    else
        input = input_org;


    // if Opencv > 2.2
    cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create("GFTT");
    featureDetector->detect(input, keypoints);

    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("BRIEF");
    descriptorExtractor->compute(input, keypoints, descriptors);

    //SIFT
//    cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create("SIFT");
//    featureDetector->detect(input, keypoints);

//    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create("SIFT");
//    descriptorExtractor->compute(input, keypoints, descriptors);

//    cout<<"#keypoints: "<<keypoints.size()<<endl;

    t1=clock()-t0;
//    cout<<"SIFT detector took: "<<(double)t1/CLOCKS_PER_SEC<<"sec"<<endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}

//sema
Mat33&
pcl::gpu::KinfuTracker::getCameraRot (int time)
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  return device_cast<Mat33> (rmats_[time]);
}
//sema
Mat33&
pcl::gpu::KinfuTracker::getCameraRotInverse (int time)
{
  if (time > (int)r_inv_.size () || time < 0)
    time = r_inv_.size () - 1;

  return r_inv_[time];
}
//sema
float3&
pcl::gpu::KinfuTracker::getCameraTrans (int time)
{
  if (time > (int)t_.size () || time < 0)
    time = t_.size () - 1;

  return t_[time];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
pcl::gpu::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume&
pcl::gpu::KinfuTracker::volume() const
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume&
pcl::gpu::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const ColorVolume&
pcl::gpu::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume&
pcl::gpu::KinfuTracker::colorVolume()
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getImage (View& view)
{

  if(isICPfail && !ishand)
      addCurrentPtCloud(view);
  else {
      Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);

      device::LightSource light;
      light.number = 1;
      light.pos[0] = device_cast<const float3>(light_source_pose);

      view.create (rows_, cols_);
      generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
  }

}

//sema
void
pcl::gpu::KinfuTracker::addCurrentPtCloud(View& view)
{
    view.create(rows_, cols_);
    raycast_view.copyTo(view);
//    view = raycast_view;
    addCurrentPointCloudOnRaycast(vmaps_curr_[0], view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  device::convert (vmaps_g_prev_[0], c);
}

//SEMA
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getCurrentFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  device::convert (vmaps_curr_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  device::convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{
  color_volume_ = pcl::gpu::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth, const View& colors, View& view, bool pause, bool roi_selected_,  View& rgb24)
{
  bool res = (*this)(depth, view, pause, roi_selected_, rgb24);

  if (res && color_volume_)
  {
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    device::Intr intr(fx_, fy_, cx_, cy_);

    Matrix3frm R_inv = rmats_.back().inverse();
    Vector3f   t     = tvecs_.back();

    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
    float3& device_tcurr = device_cast<float3> (t);

    device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0],
        colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }

  return res;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//SEMA
void
pcl::gpu::KinfuTracker::setCameraPose(Matrix3frm R, Vector3f t)
{
    int time = rmats_.size () - 1;
    cout<<"time: "<<time<<endl;

    rmats_[time] = R*init_Rcam_;
    tvecs_[time] = t + R*init_tcam_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    PCL_EXPORTS void
    paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
    {
      device::paint3DView(rgb24, view, colors_weight);
    }

    PCL_EXPORTS void
    mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
    {
      const size_t size = min(cloud.size(), normals.size());
      output.create(size);

      const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
      const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
      const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
      device::mergePointNormal(c, n, o);
    }

    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
    {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
      Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

      double rx = R(2, 1) - R(1, 2);
      double ry = R(0, 2) - R(2, 0);
      double rz = R(1, 0) - R(0, 1);

      double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
      double c = (R.trace() - 1) * 0.5;
      c = c > 1. ? 1. : c < -1. ? -1. : c;

      double theta = acos(c);

      if( s < 1e-5 )
      {
        double t;

        if( c > 0 )
          rx = ry = rz = 0;
        else
        {
          t = (R(0, 0) + 1)*0.5;
          rx = sqrt( std::max(t, 0.0) );
          t = (R(1, 1) + 1)*0.5;
          ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
          t = (R(2, 2) + 1)*0.5;
          rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

          if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
            rz = -rz;
          theta /= sqrt(rx*rx + ry*ry + rz*rz);
          rx *= theta;
          ry *= theta;
          rz *= theta;
        }
      }
      else
      {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
      }
      return Eigen::Vector3d(rx, ry, rz).cast<float>();
    }
  }
}
