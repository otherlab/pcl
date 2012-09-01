/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
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
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>

#include <pcl/console/parse.h>

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>

#include "openni_capture.h"
#include "color_handler.h"
#include "evaluation.h"

#include <pcl/common/angles.h>

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#ifdef HAVE_OPENCV  
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <pcl/gpu/utils/timers_opencv.hpp>
//#include "video_recorder.h"
typedef pcl::gpu::ScopeTimerCV ScopeTimeT;
#else
  typedef pcl::ScopeTime ScopeTimeT;
#endif

#include "../src/internal.h"

//SEMA
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/gpu/kinfu/video_recorder.h>
#include <pcl/ros/conversions.h>


using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

cv::Size rgb_cv_size(1280, 1024);
namespace pcl
{
  namespace gpu
  {
    void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
    void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms, int i) : time_ms_(time_ms), i_(i) {}
  ~SampledScopeTime()
  {
    time_ms_ += stopWatch_.getTime ();        
    if (i_ % EACH == 0 && i_)
    {
      cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << endl;
      time_ms_ = 0;        
    }
  }
private:
    StopWatch stopWatch_;
    int& time_ms_;
    int i_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{

//      Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
//      Eigen::Vector3f look_at_vector = (viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector);
//      Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
//      viewer.camera_.pos[0] = pos_vector[0]-0.4*look_at_vector[0];
//      viewer.camera_.pos[1] = pos_vector[1]-0.4*look_at_vector[1];
//      viewer.camera_.pos[2] = pos_vector[2]-0.4*look_at_vector[2];
//      viewer.camera_.focal[0] = look_at_vector[0];
//      viewer.camera_.focal[1] = look_at_vector[1];
//      viewer.camera_.focal[2] = look_at_vector[2];
//      viewer.camera_.view[0] = up_vector[0];
//      viewer.camera_.view[1] = up_vector[1];
//      viewer.camera_.view[2] = up_vector[2];
//      viewer.updateCamera ();

  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = (viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector);
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.camera_.pos[0] = pos_vector[0];
  viewer.camera_.pos[1] = pos_vector[1];
  viewer.camera_.pos[2] = pos_vector[2];
  viewer.camera_.focal[0] = look_at_vector[0];
  viewer.camera_.focal[1] = look_at_vector[1];
  viewer.camera_.focal[2] = look_at_vector[2];
  viewer.camera_.view[0] = up_vector[0];
  viewer.camera_.view[1] = up_vector[1];
  viewer.camera_.view[2] = up_vector[2];
  viewer.updateCamera ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
getViewerPose (visualization::PCLVisualizer& viewer)
{
  Eigen::Affine3f pose = viewer.getViewerPose();
  Eigen::Matrix3f rotation = pose.linear();

  Matrix3f axis_reorder;  
  axis_reorder << 0,  0,  1,
                 -1,  0,  0,
                  0, -1,  0;

  rotation = rotation * axis_reorder;
  pose.linear() = rotation;
  return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
writeCloudFile (int format, const CloudT& cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
writePoligonMeshFile (int format, const pcl::PolygonMesh& mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{    
  typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());
    
  pcl::copyPointCloud (points, *merged_ptr);
  //pcl::copyPointCloud (colors, *merged_ptr); why error?
  //pcl::concatenateFields (points, colors, *merged_ptr); why error? 
    
  for (size_t i = 0; i < colors.size (); ++i)
    merged_ptr->points[i].rgba = colors.points[i].rgba;
      
  return merged_ptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles, pcl::PointCloud<pcl::RGB> color_cloud)
{ 
  if (triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);

//  cout<<"1"<<endl;
//  pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb;
//cout<<"2"<<endl;
//  cloud_xyzrgb.points.resize(cloud.size());
//  cout<<"3"<<endl;
//  for (size_t i = 0; i < cloud.points.size(); i++) {
//      cloud_xyzrgb.points[i].x = cloud.points[i].x;
//      cloud_xyzrgb.points[i].y = cloud.points[i].y;
//      cloud_xyzrgb.points[i].z = cloud.points[i].z;

//      if(color_cloud.size() > i) {
//          cloud_xyzrgb.points[i].r = color_cloud.points[i].r;
//          cloud_xyzrgb.points[i].g = color_cloud.points[i].g;
//          cloud_xyzrgb.points[i].b = color_cloud.points[i].b;
//      }
//  }
//cout<<"4"<<endl;
  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
  pcl::toROSMsg(cloud, mesh_ptr->cloud);
//  pcl::toROSMsg(cloud_xyzrgb, mesh_ptr->cloud);
  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);              
    mesh_ptr->polygons[i] = v;
  }    
  return mesh_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
  CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

    cloud_viewer_.setBackgroundColor (0, 0, 0.15);
    cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    cloud_viewer_.addCoordinateSystem (1.0);
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.camera_.clip[0] = 0.01;
    cloud_viewer_.camera_.clip[1] = 10.01;
    cloud_viewer_.setPosition(1400,800);
  }

  void
  show (const KinfuTracker& kinfu)
  {
    kinfu.getLastFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr_->points, c);
    cloud_ptr_->width = cloud_device_.cols ();
    cloud_ptr_->height = cloud_device_.rows ();
    cloud_ptr_->is_dense = false;

    cloud_viewer_.removeAllPointClouds ();
    cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
    cloud_viewer_.spinOnce ();
  }

  void
  setViewerPose (const Eigen::Affine3f& viewer_pose) {
    ::setViewerPose (cloud_viewer_, viewer_pose);
  }

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  DeviceArray2D<PointXYZ> cloud_device_;
  visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView() : paint_image_ (false), accumulate_views_ (false)
  {
    viewerScene_.setWindowTitle ("View3D from ray tracing");
    viewerScene_.setPosition(15, 800);

    viewerDepth_.setWindowTitle ("Kinect Depth stream");
    viewerDepth_.setPosition(1200, 15);
    //viewerColor_.setWindowTitle ("Kinect RGB stream");
  }

  void
  createView(KinfuTracker& kinfu)
  {
      view_device_.create(kinfu.rows(), kinfu.cols());
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, KinfuTracker::View rgb_masked, Eigen::Affine3f* pose_ptr = 0)
  {
    view_device_.create(kinfu.rows(), kinfu.cols());
    if (pose_ptr)
    {
        raycaster_ptr_->run(kinfu.volume(), *pose_ptr, view_device_, rgb_masked);
        raycaster_ptr_->generateSceneView(view_device_);
    }
    else
      kinfu.getImage (view_device_);

    if (paint_image_ && registration && !pose_ptr)
    {
      colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
      paint3DView (colors_device_, view_device_);
    }

    int cols;
    view_device_.download (view_host_, cols);
    viewerScene_.showRGBImage ((unsigned char*)&view_host_[0], view_device_.cols (), view_device_.rows ());
    //viewerScene_.spinOnce();

    //viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);
    //viewerColor_.spinOnce();

#ifdef HAVE_OPENCV
    if (accumulate_views_)
    {
      views_.push_back (cv::Mat ());
      cv::cvtColor (cv::Mat (1024, 1280, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
      //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
    }
#endif
  }

  void
  showDepth (const PtrStepSz<const unsigned short>& depth) { viewerDepth_.showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true); }
  
  void
  showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose, DeviceArray2D<KinfuTracker::PixelRGB> rgb_masked)
  {            
    raycaster_ptr_->run(kinfu.volume(), pose, view_device_, rgb_masked);
    raycaster_ptr_->generateDepthImage(generated_depth_);    

    int c;
    vector<unsigned short> data;
    generated_depth_.download(data, c);

    viewerDepth_.showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
  }

  void
  toggleImagePaint()
  {
    paint_image_ = !paint_image_;
    cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
  }

  bool paint_image_;
  bool accumulate_views_;



  //visualization::ImageViewer viewerColor_;

  KinfuTracker::View view_device_;
  KinfuTracker::View colors_device_;
  vector<KinfuTracker::PixelRGB> view_host_;

  RayCaster::Ptr raycaster_ptr_;

  KinfuTracker::DepthMap generated_depth_;

  visualization::ImageViewer viewerDepth_;
  visualization::ImageViewer viewerScene_;
  
#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
  enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

  SceneCloudView() : extraction_mode_ (CPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false), cloud_viewer_ ("Scene Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
    normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
    combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
    point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

    cloud_viewer_.setBackgroundColor (0, 0, 0);
//    cloud_viewer_.addCoordinateSystem (1.0);//1.0
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.camera_.clip[0] = 0.01;//0.01
    cloud_viewer_.camera_.clip[1] = 10.01;

//    cloud_viewer_.addText ("H: print help", 2, 15, 20, 34, 135, 246);

    //sema
    cloud_viewer_.setSize(640, 480);


    //SEMA
    roi_added_ = false;
    editBoxDimension = 'x';
    x_min = 0.; x_max = 1;
    y_min = 0.; y_max = 1;
    z_min = 0.; z_max = 1;
  }

  void
  show (KinfuTracker& kinfu, bool integrate_colors)
  {
//      kinfu.volume().postProcess();

      viewer_pose_ = kinfu.getCameraPose();

      ScopeTimeT time ("PointCloud Extraction");
      cout << "\nGetting cloud... " << flush;

      valid_combined_ = false;

      if (extraction_mode_ != GPU_Connected6)     // So use CPU
      {
          cout<<"Use CPU"<<endl;
          kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
      }
      else
      {
          DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);

          if (compute_normals_)
          {
              kinfu.volume().fetchNormals (extracted, normals_device_);
              pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
              combined_device_.download (combined_ptr_->points);
              combined_ptr_->width = (int)combined_ptr_->points.size ();
              combined_ptr_->height = 1;

              valid_combined_ = true;
          }
          else
          {
              extracted.download (cloud_ptr_->points);
              cloud_ptr_->width = (int)cloud_ptr_->points.size ();
              cloud_ptr_->height = 1;
          }

          if (integrate_colors)
          {
              kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
              point_colors_device_.download(point_colors_ptr_->points);
              point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
              point_colors_ptr_->height = 1;
          }
          else
              point_colors_ptr_->points.clear();
      }
      size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
      cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

      cloud_viewer_.removeAllPointClouds ();
      if (valid_combined_)
      {
          visualization::PointCloudColorHandlerRGBHack<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
          cloud_viewer_.addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
          cloud_viewer_.addPointCloudNormals<PointNormal>(combined_ptr_, 50);
      }
      else
      {
          visualization::PointCloudColorHandlerRGBHack<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
          cloud_viewer_.addPointCloud<PointXYZ> (cloud_ptr_, rgb);
      }


  }

  void
  toggleCube(const Eigen::Vector3f& size)
  {
      if (cube_added_)
          cloud_viewer_.removeShape("cube");
      else
        cloud_viewer_.addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

      cube_added_ = !cube_added_;
  }

  void
  toggleExctractionMode ()
  {
    extraction_mode_ = (extraction_mode_ + 1) % 3;

    switch (extraction_mode_)
    {
    case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
    case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
    case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
    }
    ;
  }

  void
  toggleNormals ()
  {
    compute_normals_ = !compute_normals_;
    cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
  }

  void
  clearClouds (bool print_message = false)
  {
    cloud_viewer_.removeAllPointClouds ();
    cloud_ptr_->points.clear ();
    normals_ptr_->points.clear ();    
    if (print_message)
      cout << "Clouds/Meshes were cleared" << endl;
  }

  void
  showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {


    ScopeTimeT time ("Mesh Extraction");
    cout << "\nGetting mesh... " << flush;

    if (!marching_cubes_)
      marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

    DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
cout << "\nGetting mesh... " << flush;
    mesh_ptr_ = convertToMesh(triangles_device, *point_colors_ptr_);

    cloud_viewer_.removeAllPointClouds ();
    if (mesh_ptr_)
      cloud_viewer_.addPolygonMesh(*mesh_ptr_);	
    
    cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }
  void
  showMesh2(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {
//      kinfu.volume().postProcess();
      ScopeTimeT time ("Mesh Extraction");
      cout << "\nGetting mesh... " << flush;

      if (!marching_cubes_)
        marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

      DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
      mesh_ptr_ = convertToMesh(triangles_device, *point_colors_ptr_);

      cloud_viewer_.removeAllPointClouds ();
      if (mesh_ptr_)
        cloud_viewer_.addPolygonMesh(*mesh_ptr_);

      cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }

  //SEMA
  //
  void toggleROI()
  {
      if (roi_added_)
          cloud_viewer_.removeShape("roi");
      else
      {
//          cloud_viewer_.removeShape("cube");
          addBox();
      }
      roi_added_ = !roi_added_;
  }
  //add a ROI
  void addBox()
  {
//      Eigen::Vector3f size = Eigen::Vector3f(x_min,y_min,z_min);
//      size[0] = x_min; size[1] = y_min; size[2] = z_min;
//      cloud_viewer_.removeShape("ROI");
//      cloud_viewer_.addCube(size, Eigen::Quaternionf::Identity(),
//                            x_max-x_min, y_max-y_min, z_max-z_min, "roi");
        cloud_viewer_.addCube (x_min, x_max, y_min, y_max, z_min, z_max,
                             1, 0., 0., "roi", 0 );
  }
  //SEMA
  //edit ROI dimensions
  void editBoxDimensions(float d1, float d2)
  {
      switch(editBoxDimension)
      {
          case 'x':
              x_min += d1;
              x_max += d2;
              break;
          case 'y':
              y_min += d1;
              y_max += d2;
              break;
          case 'z':
              z_min += d1;
              z_max += d2;
              break;
      }

      cloud_viewer_.removeShape("roi");
      addBox();

  }
  //SEMA
  PointCloud<PointXYZ>::Ptr getROIVolume()
  {

      cloud_viewer_.removeAllPointClouds ();
      if (valid_combined_)
      {
          CropBox<PointNormal> box_crop;

          box_crop.setInputCloud(combined_ptr_);
          box_crop.setMin(Eigen::Vector4f(x_min,y_min,z_min, 0));
          box_crop.setMax(Eigen::Vector4f(x_max,y_max,z_max, 0));

          PointCloud<PointNormal>::Ptr cloud_roi_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);

          box_crop.filter(*cloud_roi_);

          visualization::PointCloudColorHandlerRGBHack<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
          cloud_viewer_.addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
          cloud_viewer_.addPointCloudNormals<PointNormal>(combined_ptr_, 50);
//          return cloud_roi_;
      }
      else
      {
          CropBox<PointXYZ> box_crop;

          box_crop.setInputCloud(cloud_ptr_);
          box_crop.setMin(Eigen::Vector4f(x_min,y_min,z_min, 0));
          box_crop.setMax(Eigen::Vector4f(x_max,y_max,z_max, 0));

          PointCloud<PointXYZ>::Ptr cloud_roi_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

          box_crop.filter(*cloud_roi_);

          visualization::PointCloudColorHandlerRGBHack<PointXYZ> rgb(cloud_roi_, point_colors_ptr_);
          cloud_viewer_.addPointCloud<PointXYZ> (cloud_roi_, rgb);

          cloud_ptr_ = cloud_roi_;

          return cloud_roi_;
      }

  }

  //SEMA
  //dimensions of ROI
  float x_min, x_max, y_min, y_max, z_min, z_max;
  char editBoxDimension;
  bool roi_added_;
  ////
    
  int extraction_mode_;
  bool compute_normals_;
  bool valid_combined_;
  bool cube_added_;

  Eigen::Affine3f viewer_pose_;

  visualization::PCLVisualizer cloud_viewer_;

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  PointCloud<Normal>::Ptr normals_ptr_;

  DeviceArray<PointXYZ> cloud_buffer_device_;
  DeviceArray<Normal> normals_device_;

  PointCloud<PointNormal>::Ptr combined_ptr_;
  DeviceArray<PointNormal> combined_device_;  

  DeviceArray<RGB> point_colors_device_; 
  PointCloud<RGB>::Ptr point_colors_ptr_;

  MarchingCubes::Ptr marching_cubes_;
  DeviceArray<PointXYZ> triangles_buffer_device_;

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
  enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_OBJ = 7, MESH_VTK = 8, MESH_PLY = 9 };
  
  KinFuApp(CaptureOpenNI& source, float vsz) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
      registration_ (false), integrate_colors_ (false), capture_ (source), pause_(false), roi_selected_(false), isfirst(true), ishand(false)
  {    
    //SEMA
    coefficients = ModelCoefficients::Ptr(new ModelCoefficients);
    coefficients->values.resize (4);    // We need 4 values
    coefficients->values[0] = 0;
    coefficients->values[1] = 1;
    coefficients->values[2] = 0;
    coefficients->values[3] = 1;

    //Init Kinfu Tracker
    Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);

    float f = capture_.depth_focal_length_VGA;
    kinfu_.setDepthIntrinsics (f, f);
    kinfu_.volume().setSize (volume_size);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
//    Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
    Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 );
//    Eigen::Vector3f t = Vector3f (vsz/2., vsz/2., 0);

    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);


    kinfu_.setInitalCameraPose (pose);
    kinfu_.volume().setTsdfTruncDist (0.002f/*meters*/);    //0.030f
    kinfu_.setIcpCorespFilteringParams (0.06f/*meters*/, sin ( pcl::deg2rad(10.f) ));//(0.2f/*meters*/, sin ( pcl::deg2rad(20.f) ))
    kinfu_.setDepthTruncationForICP(vsz+0.1/*meters*/);//5.f
    kinfu_.setCameraMovementThreshold(0.001f);//org: 0.001f
    
    //Init KinfuApp            
    tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols (), f, f) );

    scene_cloud_view_.cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);

//    scene_cloud_view_.cloud_viewer_.registerMouseCallback (mouse_callback, (void*)this);
//    scene_cloud_view_.cloud_viewer_.registerPointPickingCallback (picking_callback, (void*)this);

    image_view_.viewerDepth_.registerKeyboardCallback (keyboard_callback, (void*)this);
    image_view_.viewerScene_.registerKeyboardCallback (keyboard_callback, (void*)this);

    float diag = sqrt ((float)kinfu_.cols () * kinfu_.cols () + kinfu_.rows () * kinfu_.rows ());
//    scene_cloud_view_.cloud_viewer_.camera_.fovy = 2 * atan (diag / (2 * f)) * 1.5;
    scene_cloud_view_.cloud_viewer_.camera_.fovy = 2 * atan (kinfu_.cols () / (2 * f)) ;//* 1.5;
    
//    scene_cloud_view_.toggleCube(volume_size);

    cout<<"tsdf trunc dist: "<<kinfu_.volume().getTsdfTruncDist()<<endl;

    kinfu_.hand.reset();

  }

  ~KinFuApp()
  {
    if (evaluation_ptr_)
      evaluation_ptr_->saveAllPoses(kinfu_);
  }

  void
  initCurrentFrameView ()
  {
    current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
    current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    current_frame_cloud_view_->setViewerPose (kinfu_.getCameraPose ());
  }

  void
  tryRegistrationInit ()
  {
    registration_ = capture_.setRegistration (true);
    cout << "Registration mode: " << (registration_ ?  "On" : "Off (not supported by source)") << endl;
  }

  void 
  toggleColorIntegration(bool force = false)
  {
      cout << "Color integration ??"<<endl;
    if (registration_ || force)
    {
      const int max_color_integration_weight = -1;
      kinfu_.initColorIntegration(max_color_integration_weight);
      integrate_colors_ = true;      
    }
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off (not supported by source)") << endl;
  }

  void
  toggleIndependentCamera()
  {
    independent_camera_ = !independent_camera_;
    cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
  }
  
  void
  toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
  {
    evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
    if (!match_file.empty())
        evaluation_ptr_->setMatchFile(match_file);

    kinfu_.setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols (), 
        evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
  }



  void
  execute ()
  {
    PtrStepSz<const unsigned short> depth;
    PtrStepSz<const KinfuTracker::PixelRGB> rgb24;
    KinfuTracker::View rgb_masked;
    int time_ms = 0;
    bool has_image = false;

    kinfu_.ishand = ishand;

    for (int i = 0; !exit_; ++i)
    {


      bool has_frame = evaluation_ptr_ ? evaluation_ptr_->grab(i, depth) : capture_.grab (depth, rgb24);      
      if (!has_frame)
      {
        cout << "Can't grab" << endl;
        break;
      }

      rgb_masked.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);

      depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
      if (integrate_colors_)
          image_view_.colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);

      {
        SampledScopeTime fps(time_ms, i);
      
        //run kinfu algorithm
//        if(!pause_)//SEMA
//        {

        image_view_.createView(kinfu_);


            if (integrate_colors_)
              has_image = kinfu_ (depth_device_, image_view_.colors_device_, image_view_.view_device_, pause_, roi_selected_, rgb_masked);
            else
                has_image = kinfu_ (depth_device_, image_view_.view_device_, pause_, roi_selected_, rgb_masked);

//            coefficients = kinfu_.plane_coeffs;
//        }
      }

      if (scan_)
      {
        scan_ = false;

        //get average color from video
        device::Intr intr = kinfu_.getDepthIntrinsics();

        float3 device_volume_size = pcl::device::device_cast<const float3>(kinfu_.tsdf_volume_->getSize());

        scene_cloud_view_.show (kinfu_, true);

    //    DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);
    //    kinfu_.tsdf_volume_().fetchCloud(extracted, point_colors_device_);

        cout<<"number of points:             "<<scene_cloud_view_.cloud_ptr_->size()<<endl;

        scene_cloud_view_.point_colors_device_.create(scene_cloud_view_.cloud_ptr_->size());
        cout<<"a.."<<endl;
    //    const PointCloud<PointXYZ>* a(scene_cloud_view_.cloud_ptr_->points);
    //    DeviceArray<PointXYZ> cloud_buffer_device_(scene_cloud_view_.cloud_ptr_, scene_cloud_view_.cloud_ptr_->size());
        scene_cloud_view_.cloud_buffer_device_.upload(scene_cloud_view_.cloud_ptr_->points);
    //    DeviceArray<PointXYZ> extracted = kinfu_.volume().fetchCloud (scene_cloud_view_.cloud_buffer_device_);
    cout<<"b.."<<recorder.total_<<endl;
    //    scene_cloud_view_.cloud_buffer_device_.release();


        device::setBlack((uchar4*)scene_cloud_view_.point_colors_device_.ptr(), scene_cloud_view_.point_colors_device_.size());
        for(int i=0; i < recorder.total_ ; i++)
        {

    //        cout<<scene_cloud_view_.point_colors_device_.size()<<" 1.. "<<scene_cloud_view_.cloud_buffer_device_.size()<<endl;
cout<<i<<endl;
            DeviceArray2D<PixelRGB> kinect_rgb;
//            fillDeviceArray2D(recorder.images_[i], kinect_rgb);
            vector<KinfuTracker::PixelRGB> rgb_host_;

            for(int x=0; x<recorder.images_[i].rows; x++) {
                for(int y=0; y<recorder.images_[i].cols; y++)
                {
                    cv::Vec3b p(recorder.images_[i].at<cv::Vec3b>(x, y));
                    PixelRGB c;
                    c.b = (char)p[0];
                    c.g = (char)p[1];
                    c.r = (char)p[2];

                    rgb_host_.push_back(c);

      //              PixelRGB c = rgb_host_[x*cols+y];
      //              video_frame.at<cv::Vec3b>(x, y) = cv::Vec3b(c.b, c.g, c.r);  //BGR
                }
            }

            kinect_rgb.upload(rgb_host_, recorder.images_[i].cols);

            device::integrateColorFromRGB (kinect_rgb,  intr,
                                           device_volume_size,
                                           kinfu_.getCameraRotInverse(i),
                                           kinfu_.getCameraTrans(i),
                                           kinfu_.tsdf_volume_->getTsdfTruncDist(),
                                           kinfu_.vmaps_[i], kinfu_.nmaps_[i], scene_cloud_view_.cloud_buffer_device_, (uchar4*)scene_cloud_view_.point_colors_device_.ptr() /*scene_cloud_view_.point_colors_device_*/,
                                           scene_cloud_view_.point_colors_device_.size());

        }
        scene_cloud_view_.cloud_viewer_.removeAllPointClouds ();
        scene_cloud_view_.point_colors_device_.download(scene_cloud_view_.point_colors_ptr_->points);
        scene_cloud_view_.point_colors_ptr_->width = (int)scene_cloud_view_.point_colors_ptr_->points.size ();
        scene_cloud_view_.point_colors_ptr_->height = 1;
        visualization::PointCloudColorHandlerRGBHack<PointXYZ> rgb(scene_cloud_view_.cloud_ptr_, scene_cloud_view_.point_colors_ptr_);
        scene_cloud_view_.cloud_viewer_.addPointCloud<PointXYZ> (scene_cloud_view_.cloud_ptr_, rgb);
        scene_cloud_view_.cloud_viewer_.spinOnce ();


        integrate_colors_ = true;
        writeCloud(3);

//        scene_cloud_view_.show (kinfu_, integrate_colors_);

        if (scan_volume_)
        {
          // download tsdf volume
          {
            ScopeTimeT time ("tsdf volume download");
            cout << "Downloading TSDF volume from device ... " << flush;
            kinfu_.volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
            tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_.volume().getSize ());
            cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;
          }
          {
            ScopeTimeT time ("converting");
            cout << "Converting volume to TSDF cloud ... " << flush;
            tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
            cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;
          }
        }
        else
          cout << "[!] tsdf volume download is disabled" << endl << endl;
      }

      if (scan_mesh_)
      {
          scan_mesh_ = false;
          scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
          writeMesh(7);
//          scene_cloud_view_.showMesh2(kinfu_, integrate_colors_);
//          writeMesh(7);
      }

      if (current_frame_cloud_view_)
      {
        current_frame_cloud_view_->show (kinfu_);
      }

      if(isfirst){
          isfirst=false;
          cv::Mat video_frame (rgb_cv_size, CV_8UC3, cv::Scalar (0));
          fill_video_frame(video_frame, rgb_masked);
          recorder.push_back(video_frame);//, dept_video, video_frame);
      }
      if (has_image)// && !ishand)
      {
        Eigen::Affine3f viewer_pose = getViewerPose(scene_cloud_view_.cloud_viewer_);
        image_view_.showScene (kinfu_, rgb24, registration_, rgb_masked, independent_camera_ ? &viewer_pose : 0);
        image_view_.viewerScene_.setWindowTitle ("View3D from ray tracing"); //SEMA

        //sema
        //record into a video
//        recorder.push_back_kinfu_image(rgb_masked);

        cv::Mat video_frame (rgb_cv_size, CV_8UC3, cv::Scalar (0));
        fill_video_frame(video_frame, rgb_masked);
        recorder.push_back(video_frame);//, dept_video, video_frame);
//        if(i==0)
//            kinfu_.rgb_curr.operator =(video_frame);
//        else
//        {
//            kinfu_.rgb_prev.operator =(kinfu_.rgb_curr);
//            kinfu_.rgb_curr.operator =(video_frame);
//        }
      }

      //SEMA
      else
      {
//          Eigen::Affine3f viewer_pose = getViewerPose(scene_cloud_view_.cloud_viewer_);
          image_view_.showScene (kinfu_, rgb24, registration_, rgb_masked, 0);
          image_view_.viewerScene_.setWindowTitle ("!!!!!!!!!!!!ICP FAILED!!!!!!!!!!!!"); //SEMA
      }

      
      image_view_.showDepth (depth);
      //image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());
      
      if (!independent_camera_)
        setViewerPose (scene_cloud_view_.cloud_viewer_, kinfu_.getCameraPose());
      
      scene_cloud_view_.cloud_viewer_.spinOnce (3);



      //fancy ICP
      /*
          pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

          Eigen::Affine3f xf(kinfu_.getCameraPose());
          cout<<"cam pose: "<<endl<< xf(0,0)<<" "<< xf(0,1)<<" "<<xf(0,2)<<endl
                                  << xf(1,0)<<" "<< xf(1,1)<<" "<<xf(1,2)<<endl
                                  << xf(2,0)<<" "<< xf(2,1)<<" "<<xf(2,2)<<endl;


          if(scene_cloud_view_.cloud_ptr_->points.size() > 0 & current_frame_cloud_view_->cloud_ptr_->points.size()>0)
          {
              cout<<current_frame_cloud_view_->cloud_ptr_->points.size()<<endl;
              cout<<scene_cloud_view_.cloud_ptr_->points.size()<<endl;
              //remove NAN points from the cloud
              std::vector<int> indices;
              pcl::removeNaNFromPointCloud(*current_frame_cloud_view_->cloud_ptr_,*current_frame_cloud_view_->cloud_ptr_, indices);
//              pcl::removeNaNFromPointCloud(*scene_cloud_view_.cloud_ptr_,*scene_cloud_view_.cloud_ptr_, indices);


//              writeCloudFile (1, current_frame_cloud_view_->cloud_ptr_);


              icp.setInputCloud(current_frame_cloud_view_->cloud_ptr_);
              icp.setInputTarget(scene_cloud_view_.cloud_ptr_);

               current_frame_cloud_view_->cloud_viewer_.removeAllPointClouds ();
               current_frame_cloud_view_->cloud_viewer_.addPointCloud<PointXYZ>(current_frame_cloud_view_->cloud_ptr_);
               current_frame_cloud_view_->cloud_viewer_.addPointCloud<PointXYZ>(scene_cloud_view_.cloud_ptr_,"c2");
               current_frame_cloud_view_->cloud_viewer_.spinOnce ();



              cout<<current_frame_cloud_view_->cloud_ptr_->points.size()<<endl;
              cout<<scene_cloud_view_.cloud_ptr_->points.size()<<endl;

              pcl::PointCloud<pcl::PointXYZ> Final;
              icp.align(Final);
              std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                           icp.getFitnessScore() << std::endl;
              std::cout << icp.getFinalTransformation() << std::endl;

              Eigen::Matrix4f final_icp_xf = icp.getFinalTransformation ();

              Eigen::Matrix<float, 3, 3, Eigen::RowMajor> temp;
              temp(0,0) = final_icp_xf(0,0); temp(0,1) = final_icp_xf(0,1); temp(0,2) = final_icp_xf(0,2);
              temp(1,0) = final_icp_xf(1,0); temp(1,1) = final_icp_xf(1,1); temp(1,2) = final_icp_xf(1,2);
              temp(2,0) = final_icp_xf(2,0); temp(2,1) = final_icp_xf(2,1); temp(2,2) = final_icp_xf(2,2);

              kinfu_.setCameraPose( temp,  Eigen::Vector3f(final_icp_xf(0,3), final_icp_xf(1,3), final_icp_xf(2,3)));

//              writeCloudFile (1, scene_cloud_view_.cloud_ptr_);

          }

      }*/


      //sema
//      getBasePlane();
//      kinfu_.plane_coeffs = *coefficients;

//      if(0 && i%4==0)
//      {
////          getBasePlane();

//          current_frame_cloud_view_->cloud_viewer_.removeShape("basePlane");
//          current_frame_cloud_view_->cloud_viewer_.addPlane(*coefficients, "basePlane");

//          Eigen::Affine3f global_M = kinfu_.getCameraPose(30000);

//          float d1[12] = {1.f,0.f,0.f,
//                          0.f,1.f,0.f,
//                          0.f,0.f,1.f,
//                          0.f,0.f,0.f};
//          Eigen::Matrix<float, 4, 3> I1(d1);
//          float d2[12] = {1.f,0.f,0.f,0.f,
//                          0.f,1.f,0.f,0.f,
//                          0.f,0.f,1.f,0.f};
//          Eigen::Matrix<float, 3, 4> I2(d2);
////          cerr <<"M 3x3: "<< endl<<global_M(0,0) << " "
////                          << global_M(0,1) << " "
////                          << global_M(0,2) << endl
////                          << global_M(1,0) << " "
////                          << global_M(1,1) << " "
////                          << global_M(1,2) << endl
////                          << global_M(2,0) << " "
////                          << global_M(2,1) << " "
////                          << global_M(2,2) << " "<< endl;

//          Eigen::Matrix<float, 4, 4> M;
//          M <<global_M(0,0) , global_M(0,1) , global_M(0,2) , 0,
//              global_M(1,0) , global_M(1,1) , global_M(1,2) , 0,
//              global_M(2,0) , global_M(2,1) , global_M(2,2) , 0,
//              0             , 0             , 0             , 1;


////          cerr <<"M 4x4: "<<endl<< M(0,0) << " "<< M(0,1) << " "<< M(0,2)<< " " << M(0,3)<< endl
////              << M(1,0)<< " " << M(1,1) << " "<< M(1,2) << " "<< M(1,3)<< endl
////              << M(2,0)<< " " << M(2,1) << " "<< M(2,2) << " "<< M(2,3)<<endl
////              << M(3,0)<< " " << M(3,1) << " "<< M(3,2)<< " " << M(3,3)<< endl;


//          Eigen::Vector4f plane_coeff(coefficients->values[0], coefficients->values[1],
//                                    coefficients->values[2], coefficients->values[3]);

//          plane_coeff = M.inverse() * plane_coeff;
//          coefficients->values[0] = plane_coeff[0];
//          coefficients->values[1] = plane_coeff[1];
//          coefficients->values[2] = plane_coeff[2];
//          coefficients->values[3] = plane_coeff[3];
//          //          cerr << "Global Model coefficients: " << coefficients->values[0] << " "
//          //                                              << coefficients->values[1] << " "
//          //                                              << coefficients->values[2] << " "
//          //                                              << coefficients->values[3] << endl;

//          scene_cloud_view_.cloud_viewer_.removeShape("basePlane");
//          scene_cloud_view_.cloud_viewer_.addPlane(*coefficients, "basePlane");


//          //          kinfu_.reset();
//          //pause_ = false;
//          //          break;
//      }

    }



    saveAllKinfuPoses("outputs/cam_pos_1.txt");

    recorder.save("outputs/video.avi");

    saveDepthImages();

    cout<<"total # frames: "<< kinfu_.depth_images.size() <<endl<<flush;

    scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
    pcl::io::savePLYFile("outputs/_original.ply", *scene_cloud_view_.mesh_ptr_);
//    writeMesh(7);

    PointCloud< PointXYZ>::Ptr my_cloud;
    my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
    pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);



    //post-processing
    //pcl ICP -> reconstruct surface
    int rp=0;
    kinfu_.init_rmats_icp();
//    while(my_cloud->size()>0 && rp <2)
//    {
//        kinfu_.reconstructWithModelProxy2 (my_cloud, 1);
//        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);//get 3d mesh
//        char fnm[50];
//        sprintf(fnm, "outputs/icp_test%d.ply", rp);
//        pcl::io::savePLYFile(fnm, *scene_cloud_view_.mesh_ptr_);
////        writeMesh(7);
//        my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
//        pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);
//        char fn[50];
//        sprintf(fn, "outputs/cam_pos_2t_%04d.txt", rp);
//        saveAllKinfuPoses(fn);
//        rp++;
//    }
//    rp=0;
//    kinfu_.revert_rmats_();
    while(my_cloud->size()>0 && rp <1)
    {
        kinfu_.reconstructWithModelProxy(my_cloud, 1);
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);//get 3d mesh
        char fnm[50];
        sprintf(fnm, "outputs/icp_plane%d.ply", rp);
        pcl::io::savePLYFile(fnm, *scene_cloud_view_.mesh_ptr_);
//        writeMesh(7);
        my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
        pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);
//        char fn[50];
//        sprintf(fn, "outputs/cam_pos_2aa_%04d.txt", rp);
        saveAllKinfuPoses("outputs/cam_pos_2.txt");
        rp++;
    }

//    kinfu_.raw_depth_to_bilateraled();
//    saveDepthImages();
    return;

    rp=0;
    kinfu_.revert_rmats_();
    while(my_cloud->size()>0 && rp <2)
    {
        kinfu_.reconstructWithModelProxy(my_cloud, 0);
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);//get 3d mesh
        char fnm[50];
        sprintf(fnm, "outputs/icp_point%d.ply", rp);
        pcl::io::savePLYFile(fnm, *scene_cloud_view_.mesh_ptr_);
//        writeMesh(7);
        my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
        pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);
        char fn[50];
        sprintf(fn, "outputs/cam_pos_2ab_%04d.txt", rp);
        saveAllKinfuPoses(fn);
        rp++;
    }

    rp=0;
    kinfu_.revert_rmats_();
    while(my_cloud->size()>0 && rp <2)
    {
        kinfu_.reconstructWithModelProxy_NonLinearICP(my_cloud, 1);
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);//get 3d mesh
        char fnm[50];
        sprintf(fnm, "outputs/icp_nl_plane%d.ply", rp);
        pcl::io::savePLYFile(fnm, *scene_cloud_view_.mesh_ptr_);
//        writeMesh(7);
        my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
        pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);
        char fn[50];
        sprintf(fn, "outputs/cam_pos_2bb_%04d.txt", rp);
        saveAllKinfuPoses(fn);
        rp++;
    }

    rp=0;
    kinfu_.revert_rmats_();
    while(my_cloud->size()>0 && rp <2)
    {
        kinfu_.reconstructWithModelProxy_NonLinearICP(my_cloud, 0);
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);//get 3d mesh
        char fnm[50];
        sprintf(fnm, "outputs/icp_nl_point%d.ply", rp);
        pcl::io::savePLYFile(fnm, *scene_cloud_view_.mesh_ptr_);
//        writeMesh(7);
        my_cloud = PointCloud< PointXYZ >::Ptr (new PointCloud<PointXYZ>);
        pcl::fromROSMsg (scene_cloud_view_.mesh_ptr_->cloud, *my_cloud);
        char fn[50];
        sprintf(fn, "outputs/cam_pos_2bc_%04d.txt", rp);
        saveAllKinfuPoses(fn);
        rp++;
    }


  }

//  //SEMA
//  void fill_video_frame(cv::Mat& video_frame, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24)
//  {
//      for(int x=0; x<video_frame.rows; x++)
//          for(int y=0; y<video_frame.cols; y++)
//              video_frame.at<cv::Vec3b>(x, y) = cv::Vec3b(rgb24.ptr(x)[y].b,
//                                                          rgb24.ptr(x)[y].g,
//                                                          rgb24.ptr(x)[y].r);  //BGR
//  }
  void saveDepthImages()
  {
      for(int i=0; i<kinfu_.depth_images.size(); i++)
      {
          vector<ushort> depth_host_ = kinfu_.depth_images[i];
          vector<ushort> depth_raw_host_ = kinfu_.depth_raw_images[i];

          cv::Mat frame, frame_raw;
          frame.create (cv::Size (640, 480), CV_16U);
          frame_raw.create (cv::Size (640, 480), CV_16U);

          for(int x=0; x<frame.rows; x++)
              for(int y=0; y<frame.cols; y++)
              {
                  frame.at< ushort >(x, y) = depth_host_[x*frame.cols+y];
                  frame_raw.at< ushort >(x, y) = depth_raw_host_[x*frame_raw.cols+y];
              }

          std::vector<int> compression_params;
          compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
          compression_params.push_back(0);

          char fn[50];
          sprintf(fn, "outputs/depth/depth_%04d.png", i);
          char fn_raw[50];
          sprintf(fn_raw, "outputs/depth_raw/depth_%04d.png", i);
          try {
              cv::imwrite(fn, frame, compression_params);
              cv::imwrite(fn_raw, frame_raw, compression_params);
          }
          catch (std::runtime_error& ex) {
              fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
              return ;
          }

      }

  }

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
  void fillDeviceArray2D(cv::Mat video_frame, KinfuTracker::View& rgb24)
  {
//      int cols;
      vector<KinfuTracker::PixelRGB> rgb_host_;
//      rgb24.download (rgb_host_, cols);

      for(int x=0; x<video_frame.rows; x++)
          for(int y=0; y<video_frame.cols; y++)
          {
              cv::Vec3b p(video_frame.at<cv::Vec3b>(x, y));
              PixelRGB c;
              c.b = p[0];
              c.g = p[1];
              c.r = p[2];

              rgb_host_.push_back(c);

//              PixelRGB c = rgb_host_[x*cols+y];
//              video_frame.at<cv::Vec3b>(x, y) = cv::Vec3b(c.b, c.g, c.r);  //BGR
          }

      rgb24.upload(rgb_host_, 1);
  }

  //SEMA
  void saveAllKinfuPoses(char* logfile,  int frame_number=-1)
  {
      frame_number = kinfu_.getNumberOfPoses();

      cout << "Writing " << frame_number << " poses to " << logfile << endl;

      ofstream path_file_stream(logfile);
      path_file_stream.setf(ios::fixed,ios::floatfield);

      path_file_stream<<kinfu_.getDepthIntrinsics().cx<<" "<<kinfu_.getDepthIntrinsics().cy<<" "<<kinfu_.getDepthIntrinsics().fx<<" "<<kinfu_.getDepthIntrinsics().fy<<endl;
      for(int i = 0; i < frame_number; ++i)
      {
          Eigen::Affine3f pose = kinfu_.getCameraPose(i);

          //        Eigen::Quaternionf q(pose.rotation());
          //        Eigen::Vector3f t = pose.translation();
          //        path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
          //        path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
          
          Eigen::Matrix3f R(pose.rotation());
          Eigen::Vector3f T(pose.translation());
          
          path_file_stream<< R.coeff(0,0) <<" "<< R.coeff(0,1) <<" "<< R.coeff(0,2) <<" "<< T.coeff(0) <<" ";
          path_file_stream<< R.coeff(1,0) <<" "<< R.coeff(1,1) <<" "<< R.coeff(1,2) <<" "<< T.coeff(1) <<" ";
          path_file_stream<< R.coeff(2,0) <<" "<< R.coeff(2,1) <<" "<< R.coeff(2,2) <<" "<< T.coeff(2) <<endl;


          //      setViewerPose (scene_cloud_view_.cloud_viewer_, pose);
          //      char fn2[50];
          //      sprintf(fn2, "video/model%d.png", i);
          //      cout<<fn2<<endl;
          //      scene_cloud_view_.cloud_viewer_.saveScreenshot(fn2);
          
      }
    path_file_stream.close();
  }

  //SEMA
  void ROI()
  {
//      PointCloud<PointXYZ>::Ptr cloud_roi_ = scene_cloud_view_.getROIVolume();

//      getBasePlane(cloud_roi_);

//      removeBasePlane(cloud_roi_);

//      scene_cloud_view_.cloud_viewer_.removeShape("basePlane");
//      scene_cloud_view_.cloud_viewer_.addPlane(*coefficients, "basePlane");

      ModelCoefficients::Ptr box_boundaries = ModelCoefficients::Ptr(new ModelCoefficients);
      box_boundaries->values.resize (6);    // We need 6 values
      box_boundaries->values[0] = scene_cloud_view_.x_min;
      box_boundaries->values[1] = scene_cloud_view_.x_max;
      box_boundaries->values[2] = scene_cloud_view_.y_min;
      box_boundaries->values[3] = scene_cloud_view_.y_max;
      box_boundaries->values[4] = scene_cloud_view_.z_min;
      box_boundaries->values[5] = scene_cloud_view_.z_max;

      kinfu_.setROI(box_boundaries);

      kinfu_.reduceTsdfWeights(box_boundaries);

//      kinfu_.extractObject(box_boundaries, coefficients);

      roi_selected_ = true;

  }
  void removeBasePlane ( PointCloud<PointXYZ>::Ptr cloud_ptr)
  {
      float eps = 0.006; // in meters

      boost::shared_ptr<std::vector<int> > indices(new std::vector<int>());
//cout<<"sema "<<cloud_ptr->size()<<endl;
      for(int i=0; i<cloud_ptr->size(); i++)
      {
          float vx = cloud_ptr->at(i).x;
          float vy = cloud_ptr->at(i).y;
          float vz = cloud_ptr->at(i).z;

          float eval = coefficients->values[0]*vx + coefficients->values[1]*vy +
                       coefficients->values[2]*vz + coefficients->values[3];

          if(eval > eps)
          {
              indices->push_back(i);
//              cloud_ptr->erase(cloud_ptr->begin()+i);
//              i--;
          }

      }
//      cout<<"sema "<<indices->size()<<endl;

      pcl::ExtractIndices<PointXYZ> eifilter; // Initializing with true will allow us to extract the removed indices
      eifilter.setInputCloud (cloud_ptr);
       eifilter.setIndices (indices);
       PointCloud<PointXYZ> cloud_out;
       eifilter.filter (cloud_out);

       cloud_ptr->swap(cloud_out);
//       cout<<"sema "<<cloud_ptr->size()<<endl;

       scene_cloud_view_.cloud_viewer_.removeAllPointClouds ();
       scene_cloud_view_.cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr);
       scene_cloud_view_.cloud_viewer_.spinOnce ();
  }

  void getBasePlane ( PointCloud<PointXYZ>::Ptr cloud_ptr)
  {
      coefficients->values[0] = 0;
      coefficients->values[1] = 1;
      coefficients->values[2] = 0;
      coefficients->values[3] = 1000;

      //      DeviceArray2D<pcl::PointXYZ> cloud_device_;

      //      getCurrentFrameCloud (cloud_device_);

      //      int c;
      //      cloud_device_.download (cloud_ptr->points, c);
      //      cloud_ptr->width = cloud_device_.cols ();
      //      cloud_ptr->height = cloud_device_.rows ();
      //      cloud_ptr->is_dense = false;

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
      seg.segment (*inliers, *coefficients);


      if (inliers->indices.size () == 0)
      {
          PCL_ERROR ("Could not estimate a planar model for the given dataset.");
          return;
      }

      cerr << "Model coefficients: " << coefficients->values[0] << " "
           << coefficients->values[1] << " "
           << coefficients->values[2] << " "
           << coefficients->values[3] << endl;

      cerr << "Model inliers: " << inliers->indices.size () << endl;

  }



  void
  writeCloud (int format) const
  {      
    const SceneCloudView& view = scene_cloud_view_;

    if(view.point_colors_ptr_->points.empty()) // no colors
    {
      if (view.valid_combined_)
        writeCloudFile (format, view.combined_ptr_);
      else
        writeCloudFile (format, view.cloud_ptr_);
    }
    else
    {        
      if (view.valid_combined_)
        writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
      else
        writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
    }
  }

  void
  writeMesh(int format) const
  {
    if (scene_cloud_view_.mesh_ptr_) 
      writePoligonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
  }

  void
  printHelp ()
  {
    cout << endl;
    cout << "KinFu app hotkeys" << endl;
    cout << "=================" << endl;
    cout << "    H    : print this help" << endl;
    cout << "   Esc   : exit" << endl;
    cout << "    T    : take cloud" << endl;
    cout << "    A    : take mesh" << endl;
    cout << "    M    : toggle cloud exctraction mode" << endl;
    cout << "    N    : toggle normals exctraction" << endl;
    cout << "    I    : toggle independent camera mode" << endl;
    cout << "    B    : toggle volume bounds" << endl;
    cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
    cout << "    C    : clear clouds" << endl;    
    cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
    cout << "    7,8  : save mesh to PLY, VTK" << endl;
    cout << "   J, V  : TSDF volume utility" << endl;
    //SEMA
    cout << "    R    : reset the volume" << endl;
    cout << "    P    : pause/resume reconstruction" << endl;
    cout << "    Space Bar  : finish getting hand statistics" << endl;
    cout << endl;
  }  

  //SEMA
  bool roi_selected_ ;
  bool pause_;
  ModelCoefficients::Ptr coefficients;
  //video recorder
  pcl::BufferedRecorder recorder;
  bool isfirst;
  bool ishand;

  bool exit_;
  bool scan_;
  bool scan_mesh_;
  bool scan_volume_;

  bool independent_camera_;

  bool registration_;
  bool integrate_colors_;
  
  CaptureOpenNI& capture_;
  KinfuTracker kinfu_;


  boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

  KinfuTracker::DepthMap depth_device_;

  pcl::TSDFVolume<float, short> tsdf_volume_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

  SceneCloudView scene_cloud_view_;
  ImageView image_view_;

  Evaluation::Ptr evaluation_ptr_;

  static void
  keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
  {
    KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

    int key = e.getKeyCode ();

    if (e.keyUp ())
      switch (key)
      {
      case 27: app->exit_ = true; break;
      case (int)'p': case (int)'P': app->pause_ = !app->pause_;
//          if(app->pause_)   app->scan_ = true;
          if(app->pause_)  app->scan_mesh_ = true;
//          app->toggleIndependentCamera ();
          app->scene_cloud_view_.toggleROI();
          break;  //SEMA
      case (int)'r': case (int)'R': app->kinfu_.reset();
                                    app->recorder.reset();
                                    app->isfirst = true; break;  //SEMA
      case (int)'s': case (int)'S': app->scene_cloud_view_.toggleROI(); break;  //SEMA

      case (int)'l': case (int)'L': app->kinfu_.stabilizeCameraPosition();                  break;  //SEMA

      case (int)'-' : app->scene_cloud_view_.editBoxDimensions(-.05, 0); break;  //SEMA
      case (int)'.' : app->scene_cloud_view_.editBoxDimensions(0, +.05); break;  //SEMA
      case (int)'+' : app->scene_cloud_view_.editBoxDimensions(+.05, 0); break;  //SEMA
      case (int)',' : app->scene_cloud_view_.editBoxDimensions(0, -.05); break;  //SEMA

      case (int)'e' : app->ROI(); break;  //SEMA

      case (int)'x': case (int)'X': app->scene_cloud_view_.editBoxDimension = 'x';    break;  //SEMA
      case (int)'y': case (int)'Y': app->scene_cloud_view_.editBoxDimension = 'y';    break;  //SEMA
      case (int)'z': case (int)'Z': app->scene_cloud_view_.editBoxDimension = 'z';    break;  //SEMA

      //SEMA
      case (int)'k': case (int)'K':
          {
            int f=0;
              cout<<"Please enter a frame number between 0 and "<<app->kinfu_.getNumberOfPoses()<<endl;
              cin>>f;
              if(f<0 || f>app->kinfu_.getNumberOfPoses())   f=0;
              setViewerPose (app->scene_cloud_view_.cloud_viewer_, app->kinfu_.getCameraPose(f));

//              cv::Mat video_frame (cv::Size (640, 480), CV_8UC3, cv::Scalar (0));
//              app->fill_video_frame(video_frame, app->image_view_.view_device_);
//              app->recorder.push_back(video_frame);//, dept_video, video_frame);

              break;
          }

          //sema
      case (int)' ': app->ishand = false;
                     app->kinfu_.ishand = false;
                     app->kinfu_.hand.calculateModel();
                     cout<<flush;
                     break;  //SEMA


      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
      case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExctractionMode (); break;
      case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;      
      case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
      case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
      case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_.volume().getSize()); break;
      case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;
      case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;      
      case '*': app->image_view_.toggleImagePaint (); break;

//      case (int)'j': case (int)'J':
//        app->scan_volume_ = !app->scan_volume_;
//        cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
//        break;
      case (int)'v': case (int)'V':
        cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
        app->tsdf_volume_.save ("tsdf_volume.dat", true);
        cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
        cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
        pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
        cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;
        break;

      default:
        break;
      }    
  }

  //SEMA
  static void
  mouse_callback (const visualization::MouseEvent &e, void *cookie)
  {
      KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

//      if(app->scene_cloud_view_.roi_added_)
//          cout<<"hobaaa"<<endl;
  }
  //SEMA
  static void
  picking_callback (const visualization::PointPickingEvent &e, void *cookie)
  {
      KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

//      if(app->scene_cloud_view_.roi_added_)
          cout<<"point"<<endl;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//sema
string output_fn = "cloud";

template<typename CloudPtr> void
writeCloudFile (int format, const CloudPtr& cloud_prt)
{
  // filename is timestamp
  time_t t;
  time (&t);
  string fn_root = "outputs/";
  fn_root.append(ctime(&t));

//  mkdir("kinfu_outputs", 777);

  if (format == KinFuApp::PCD_BIN)
  {
    string fn = fn_root;
    fn.append(".pcd");
    cout << "Saving point cloud to '"<<fn<<"' (binary)... " << flush;
    pcl::io::savePCDFile (fn, *cloud_prt, true);
  }
  else
  if (format == KinFuApp::PCD_ASCII)
  {
      string fn = fn_root;
      fn.append(".pcd");
    cout << "Saving point cloud to '"<<fn<<"' (ASCII)... " << flush;
    pcl::io::savePCDFile (fn, *cloud_prt, false);
  }
  else   /* if (format == KinFuApp::PLY) */
  {
    string fn = fn_root;
    fn.append(".ply");
    cout << "Saving point cloud to '"<<fn<<"' (ASCII)... " << flush;
    pcl::io::savePLYFileASCII (fn, *cloud_prt);
  
  }
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePoligonMeshFile (int format, const pcl::PolygonMesh& mesh)
{
    // filename is timestamp
    time_t t;
    time (&t);
    string fn_root = "outputs/";
    fn_root.append(ctime(&t));

    if (format == KinFuApp::MESH_OBJ)
    {
        string fn = fn_root;
        fn.append(".obj");
        cout << "Saving mesh to to '"<<fn<<"'... " << flush;
      pcl::io::saveOBJFile(fn, mesh);
    }
    else if (format == KinFuApp::MESH_PLY)
  {
      string fn = fn_root;
      fn.append(".ply");
      cout << "Saving mesh to to '"<<fn<<"'... " << flush;
    pcl::io::savePLYFile(fn, mesh);
  }
  else /* if (format == KinFuApp::MESH_VTK) */
  {
      string fn = fn_root;
      fn.append(".vtk");
    cout << "Saving mesh to to '"<<fn<<"'... " << flush;
    pcl::io::saveVTKFile(fn, mesh);
  }  
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
  cout << "\nKinfu app concole parameters help:" << endl;
  cout << "    --help, -h                      : print this message" << endl;  
  cout << "    --registration, -r              : try to enable registration ( requires source to support this )" << endl;
  cout << "    --current-cloud, -cc            : show current frame cloud" << endl;
  cout << "    --save-views, -sv               : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;  
  cout << "    --registration, -r              : enable registration mode" << endl; 
  cout << "    --integrate-colors, -icf        : enable color integration mode ( allows to get cloud with colors )" << endl;
  cout << "    -volume_size <size_in_meters>   : define integration volume size" << endl;
  cout << "    -dev <deivce>, -oni <oni_file>  : select depth source. Default will be selected if not specified" << endl;
  cout << "";
  cout << " For RGBD benchmark (Requires OpenCV):" << endl; 
  cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;

  cout << "    --filename, -fn                 : set output file name" << endl;
  cout << "    --enablehandremoval, -ehr       : set output file name" << endl;
    
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
main (int argc, char* argv[])
{  
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_cli_help ();

  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);

  if(checkIfPreFermiGPU(device))
    return cout << endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << endl, 1;

  CaptureOpenNI capture;
  
  int openni_device = 0;
  std::string oni_file, eval_folder, match_file;
  if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
  {
    capture.open (openni_device);
  }
  else
  if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
  {
    capture.open (oni_file);
  }
  else
  if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
  {
    //init data source latter
    pc::parse_argument (argc, argv, "-match_file", match_file);
  }
  else
  {
    capture.open (openni_device);
    //capture.open("d:/onis/20111013-224932.oni");
    //capture.open("d:/onis/reg20111229-180846.oni");
    //capture.open("d:/onis/white1.oni");
    //capture.open("/media/Main/onis/20111013-224932.oni");
    //capture.open("20111013-225218.oni");
    //capture.open("d:/onis/20111013-224551.oni");
    //capture.open("d:/onis/20111013-224719.oni");
  }

  float volume_size = 3.f;
  pc::parse_argument (argc, argv, "-volume_size", volume_size);

  KinFuApp app (capture, volume_size);

  //sema: set output filename
  pc::parse_argument (argc, argv, "--filename", output_fn);
  pc::parse_argument (argc, argv, "-fn", output_fn);

  if (pc::find_switch (argc, argv, "-enablehandremoval"))
      app.ishand = true;
  if (pc::find_switch (argc, argv, "-ehr"))
      app.ishand = true;

  if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
    app.toggleEvaluationMode(eval_folder, match_file);

  if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
    app.initCurrentFrameView ();

  if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
    app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  
    
  if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))
      app.tryRegistrationInit();
      
  bool force = pc::find_switch (argc, argv, "-icf");
  if (force || pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))
    app.toggleColorIntegration(force);

  
  // executing
  try { app.execute (); }
  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
  catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

#ifdef HAVE_OPENCV
  for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
  {
    if (t == 0)
    {
      cout << "Saving depth map of first view." << endl;
      cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
      cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
    }
    char buf[4096];
    sprintf (buf, "./%06d.png", (int)t);
    cv::imwrite (buf, app.image_view_.views_[t]);
    printf ("writing: %s\n", buf);
  }
#endif
  return 0;
}
