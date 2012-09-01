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

#include <pcl/gpu/kinfu/tsdf_volume.h>
#include "internal.h"
#include <algorithm>
#include <Eigen/Core>

using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
using pcl::device::device_cast;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::TsdfVolume::TsdfVolume(const Vector3i& resolution) : resolution_(resolution)
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  volume_.create (volume_y * volume_z, volume_x);
  
  const Vector3f default_volume_size = Vector3f::Constant (3.f); //meters
  const float    default_tranc_dist  = 0.03f; //meters

  setSize(default_volume_size);
  setTsdfTruncDist(default_tranc_dist);

  reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::setSize(const Vector3f& size)
{  
  size_ = size;
  setTsdfTruncDist(tranc_dist_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::setTsdfTruncDist (float distance)
{
  float cx = size_(0) / resolution_(0);
  float cy = size_(1) / resolution_(1);
  float cz = size_(2) / resolution_(2);

  tranc_dist_ = std::max (distance, 2.1f * std::max (cx, std::max (cy, cz)));  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::DeviceArray2D<int> 
pcl::gpu::TsdfVolume::data() const
{
  return volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f&
pcl::gpu::TsdfVolume::getSize() const
{
    return size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3i&
pcl::gpu::TsdfVolume::getResolution() const
{
  return resolution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f
pcl::gpu::TsdfVolume::getVoxelSize() const
{    
  return size_.array () / resolution_.array().cast<float>();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float
pcl::gpu::TsdfVolume::getTsdfTruncDist () const
{
  return tranc_dist_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void 
pcl::gpu::TsdfVolume::reset()
{
  device::initVolume(volume_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchCloudHost (PointCloud<PointType>& cloud, bool connected26) const
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  int cols;
  std::vector<int> volume_host;
  volume_.download (volume_host, cols);

  cloud.points.clear ();
  cloud.points.reserve (10000);

  const int DIVISOR = device::DIVISOR; // SHRT_MAX;

#define FETCH(x, y, z) volume_host[(x) + (y) * volume_x + (z) * volume_y * volume_x]

  Array3f cell_size = getVoxelSize();

  for (int x = 1; x < volume_x-1; ++x)
  {
    for (int y = 1; y < volume_y-1; ++y)
    {
      for (int z = 0; z < volume_z-1; ++z)
      {
        int tmp = FETCH (x, y, z);
        int W = reinterpret_cast<short2*>(&tmp)->y;
        int F = reinterpret_cast<short2*>(&tmp)->x;

        if (W == 0 || F == DIVISOR)
          continue;

        Vector3f V = ((Array3f(x, y, z) + 0.5f) * cell_size).matrix ();
//        std::cout<<x<<" "<<y<<" "<<z<<std::endl;
//        std::cout<<V[0]<<std::endl;
//        std::cout<<V[1]<<std::endl;
//        std::cout<<V[2]<<std::endl;

        if (connected26)
        {
          int dz = 1;
          for (int dy = -1; dy < 2; ++dy)
            for (int dx = -1; dx < 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);

                cloud.points.push_back (xyz);
              }
            }
          dz = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = -1; dx < dy * 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * abs(Fn) + Vn * abs(F))/(abs(F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);
//                std::cout<<xyz.x<<std::endl;
//                std::cout<<xyz.y<<std::endl;
//                std::cout<<xyz.z<<std::endl<<std::endl;

                cloud.points.push_back (xyz);
              }
            }
        }
        else /* if (connected26) */
        {
          for (int i = 0; i < 3; ++i)
          {
            int ds[] = {0, 0, 0};
            ds[i] = 1;

            int dx = ds[0];
            int dy = ds[1];
            int dz = ds[2];

            int tmp = FETCH (x+dx, y+dy, z+dz);

            int Wn = reinterpret_cast<short2*>(&tmp)->y;
            int Fn = reinterpret_cast<short2*>(&tmp)->x;
            if (Wn == 0 || Fn == DIVISOR)
              continue;

            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
            {
              Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
              Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

              pcl::PointXYZ xyz;
              xyz.x = point (0);
              xyz.y = point (1);
              xyz.z = point (2);

//              std::cout<<xyz.x<<std::endl;
//              std::cout<<xyz.y<<std::endl;
//              std::cout<<xyz.z<<std::endl<<std::endl;

              cloud.points.push_back (xyz);
            }
          }
        } /* if (connected26) */
      }
    }
  }
#undef FETCH
  cloud.width  = (int)cloud.points.size ();
  cloud.height = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::DeviceArray<pcl::gpu::TsdfVolume::PointType>
pcl::gpu::TsdfVolume::fetchCloud (DeviceArray<PointType>& cloud_buffer) const
{
  if (cloud_buffer.empty ())
    cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

  float3 device_volume_size = device_cast<const float3> (size_);
  size_t size = device::extractCloud (volume_, device_volume_size, cloud_buffer);
  return (DeviceArray<PointType> (cloud_buffer.ptr (), size));
}

//sema
void
pcl::gpu::TsdfVolume::postProcess ()
{
   device::postProcessTsdf (volume_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<PointType>& normals) const
{
  normals.create (cloud.size ());
  const float3 device_volume_size = device_cast<const float3> (size_);
  device::extractNormals (volume_, device_volume_size, cloud, (device::PointType*)normals.ptr ());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<NormalType>& normals) const
{
  normals.create (cloud.size ());
  const float3 device_volume_size = device_cast<const float3> (size_);
  device::extractNormals (volume_, device_volume_size, cloud, (device::float8*)normals.ptr ());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::downloadTsdf (std::vector<float>& tsdf) const
{
  tsdf.resize (volume_.cols() * volume_.rows());
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

#pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    float tmp = reinterpret_cast<short2*>(&tsdf[i])->x;
    tsdf[i] = tmp/device::DIVISOR;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::downloadTsdfAndWeighs (std::vector<float>& tsdf, std::vector<short>& weights) const
{
  int volumeSize = volume_.cols() * volume_.rows();
  tsdf.resize (volumeSize);
  weights.resize (volumeSize);
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

#pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
    tsdf[i] = (float)(elem.x)/device::DIVISOR;    
    weights[i] = (short)(elem.y);    
  }
}

//SEMA
void
pcl::gpu::TsdfVolume::reduceTsdfByROI (pcl::ModelCoefficients::Ptr box_boundaries)
{

  device::reduceVolumeWeights(volume_, box_boundaries);
  return;
}
void
pcl::gpu::TsdfVolume::cleanTsdfByROIandPlane (pcl::ModelCoefficients::Ptr box_boundaries,
                                                      pcl::ModelCoefficients::Ptr plane_coeffs)
{


//    device::initVolume(volume_);

  device::cutVolume(volume_, box_boundaries, plane_coeffs);
  return;


//  int volume_x = resolution_(0);
//  int volume_y = resolution_(1);
//  int volume_z = resolution_(2);

//  DeviceArray2D<int> volume_2(volume_);
//  std::cout<<"b "<<volume_temp.cols()<<std::endl;
//  std::cout<<"b "<<volume_temp.rows()<<std::endl;

//  std::cout<<"b "<<volume_.cols()<<std::endl;
//  device::initVolume(volume_2);
//  std::cout<<"a"<<std::endl;
//    int cols;
//    std::vector<int> volume_host;
//    volume_.download (volume_host, cols);

//    std::vector<int> volume_temp;
//    device::initVolume(volume_2);
//    volume_2.download (volume_temp, cols);

//#define FETCH1(x, y, z) volume_temp[(x) + (y) * volume_x + (z) * volume_y * volume_x]
//#define FETCH2(x, y, z) volume_host[(x) + (y) * volume_x + (z) * volume_y * volume_x]

//  Array3f cell_size = getVoxelSize();
//std::cout<<"b "<<cloud_roi.size()<<std::endl;
//  for(int i=0; i<cloud_roi.size(); i++)
//  {
//      pcl::PointXYZ point = cloud_roi.at(i);
//      int x = round(point.x/cell_size[0]);
//      int y = round(point.y/cell_size[1]);
//      int z = round(point.z/cell_size[2]);


//      int tmp1 = FETCH1 (x, y, z);
//      int tmp2 = FETCH2 (x, y, z);
//volume_host[(x) + (y) * volume_x + (z) * volume_y * volume_x] = -100;

// reinterpret_cast<short2*>(&tmp2)->y = 0;
//       reinterpret_cast<short2*>(&tmp2)->x = -100;
//      reinterpret_cast<short2*>(&tmp1)->y = 0;//reinterpret_cast<short2*>(&tmp2)->y;
//      reinterpret_cast<short2*>(&tmp1)->x = 0;//reinterpret_cast<short2*>(&tmp2)->x;
//  }
//  volume_.upload(volume_host, cols);
//  int cols;
//  std::vector<int> volume_host;
//  volume_.download (volume_host, cols);

//  cloud.points.clear ();
//  cloud.points.reserve (10000);

//  const int DIVISOR = device::DIVISOR; // SHRT_MAX;




//  for (int x = 1; x < volume_x-1; ++x)
//  {
//    for (int y = 1; y < volume_y-1; ++y)
//    {
//      for (int z = 0; z < volume_z-1; ++z)
//      {
//        int tmp = FETCH (x, y, z);
//        int W = reinterpret_cast<short2*>(&tmp)->y;
//        int F = reinterpret_cast<short2*>(&tmp)->x;

//        if (W == 0 || F == DIVISOR)
//          continue;

//        Vector3f V = ((Array3f(x, y, z) + 0.5f) * cell_size).matrix ();

//        if (connected26)
//        {
//          int dz = 1;
//          for (int dy = -1; dy < 2; ++dy)
//            for (int dx = -1; dx < 2; ++dx)
//            {
//              int tmp = FETCH (x+dx, y+dy, z+dz);

//              int Wn = reinterpret_cast<short2*>(&tmp)->y;
//              int Fn = reinterpret_cast<short2*>(&tmp)->x;
//              if (Wn == 0 || Fn == DIVISOR)
//                continue;

//              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
//              {
//                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
//                Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

//                pcl::PointXYZ xyz;
//                xyz.x = point (0);
//                xyz.y = point (1);
//                xyz.z = point (2);

//                cloud.points.push_back (xyz);
//              }
//            }
//          dz = 0;
//          for (int dy = 0; dy < 2; ++dy)
//            for (int dx = -1; dx < dy * 2; ++dx)
//            {
//              int tmp = FETCH (x+dx, y+dy, z+dz);

//              int Wn = reinterpret_cast<short2*>(&tmp)->y;
//              int Fn = reinterpret_cast<short2*>(&tmp)->x;
//              if (Wn == 0 || Fn == DIVISOR)
//                continue;

//              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
//              {
//                Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
//                Vector3f point = (V * abs(Fn) + Vn * abs(F))/(abs(F) + abs (Fn));

//                pcl::PointXYZ xyz;
//                xyz.x = point (0);
//                xyz.y = point (1);
//                xyz.z = point (2);

//                cloud.points.push_back (xyz);
//              }
//            }
//        }
//        else /* if (connected26) */
//        {
//          for (int i = 0; i < 3; ++i)
//          {
//            int ds[] = {0, 0, 0};
//            ds[i] = 1;

//            int dx = ds[0];
//            int dy = ds[1];
//            int dz = ds[2];

//            int tmp = FETCH (x+dx, y+dy, z+dz);

//            int Wn = reinterpret_cast<short2*>(&tmp)->y;
//            int Fn = reinterpret_cast<short2*>(&tmp)->x;
//            if (Wn == 0 || Fn == DIVISOR)
//              continue;

//            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
//            {
//              Vector3f Vn = ((Array3f (x+dx, y+dy, z+dz) + 0.5f) * cell_size).matrix ();
//              Vector3f point = (V * abs (Fn) + Vn * abs (F)) / (abs (F) + abs (Fn));

//              pcl::PointXYZ xyz;
//              xyz.x = point (0);
//              xyz.y = point (1);
//              xyz.z = point (2);

//              cloud.points.push_back (xyz);
//            }
//          }
//        } /* if (connected26) */
//      }
//    }
//  }
//#undef FETCH
//  cloud.width  = (int)cloud.points.size ();
//  cloud.height = 1;
}
