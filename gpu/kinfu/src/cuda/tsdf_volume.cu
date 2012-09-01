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

#include "device.hpp"


using namespace pcl::device;

namespace pcl
{
  namespace device
  {
    template<typename T>
    __global__ void
    initializeVolume (PtrStep<T> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;


      if (x < VOLUME_X && y < VOLUME_Y)
      {
          T *pos = volume.ptr(y) + x;
          int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
          for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
             pack_tsdf (0.f, 0, *pos);
      }
    }

    //SEMA
    //a,b,c,d are plane coefficients, the rest is the boundary of the box
    template<typename T>
    __global__ void
    cut_Volume (PtrStep<T> volume, float x_min, float x_max, float y_min, float y_max,
                float z_min, float z_max, float a, float b, float c, float d)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

//      float eps = 0.006;

      T *pos = volume.ptr(y) + x;
      int z_step = VOLUME_Y * volume.step / sizeof(*pos);

      //x-dimension
      if (x < x_min && y < VOLUME_Y)
      {
#pragma unroll
          for(int z = 0; z < VOLUME_Z ; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);
      }
      else if (x > x_max && y < VOLUME_Y)
      {
#pragma unroll
          for(int z = 0; z < VOLUME_Z ; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);
      }
//      else if(x < VOLUME_X && y < VOLUME_Y)
//      {
//          //plane removal
//#pragma unroll
//          for(int z = 0; z < VOLUME_Z ; ++z, pos+=z_step)
//          {
//              float eval = a*x + b*y + c*z + d;
//              if(eval < eps)
//                  pack_tsdf (0.f, 0, *pos);
//          }
//      }

      pos = volume.ptr(y) + x;
      //y-dimension
      if (y < y_min && x < VOLUME_X)
      {
#pragma unroll
          for(int z = 0; z < VOLUME_Z ; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);
      }
      else if (y > y_max && x < VOLUME_X)
      {
#pragma unroll
          for(int z = 0; z < VOLUME_Z ; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);
      }

      //z-dimension
      pos = volume.ptr(y) + x;
      if (x < VOLUME_X && y < VOLUME_Y)
      {
#pragma unroll
          for(int z = 0; z < z_min; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);

          pos = volume.ptr(y) + x;
          pos += z_step*((int)z_max);
#pragma unroll
          for(int z = z_max; z < VOLUME_Z ; ++z, pos+=z_step)
                  pack_tsdf (0.f, 0, *pos);
      }
    }

    //SEMA
    __device__ float sqr(float a){return a*a;}

  //SEMA
  //a,b,c,d are plane coefficients, the rest is the boundary of the box
  template<typename T>
  __global__ void
  reduce_Volume_Weights (PtrStep<T> volume, float x_min, float x_max, float y_min, float y_max,
              float z_min, float z_max)
  {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    T *pos = volume.ptr(y) + x;
    int z_step = VOLUME_Y * volume.step / sizeof(*pos);

    bool is_inside = false;
     if (x > x_min && x < x_max && y > y_min && y < y_max)
         is_inside = true;


#pragma unroll
        for(int z = 0; z < VOLUME_Z; ++z, pos+=z_step)
        {
            if (is_inside && z > z_min && z < z_max)
              continue;

            //SEMA
            //distance to the ROI
            float d = 0.;
            if (is_inside)
                d = min(fabs(z-z_min), fabs(z-z_max));
            else if(z > z_min && z < z_max && y > y_min && y < y_max)
                d = min(fabs(x-x_min), fabs(x-x_max));
            else if(z > z_min && z < z_max && x > x_min && x < x_max)
                d = min(fabs(y-y_min), fabs(y-y_max));
            else if(z > z_min && z < z_max)
                d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) +
                        sqr(min(fabs(y-y_min), fabs(y-y_max))) );
            else if(x > x_min && x < x_max)
                d = sqrt( sqr(min(fabs(z-z_min), fabs(z-z_max))) +
                        sqr(min(fabs(y-y_min), fabs(y-y_max))) );
            else if(y > y_min && y < y_max)
                d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) +
                        sqr(min(fabs(z-z_min), fabs(z-z_max))) );
            else
                d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) + sqr(min(fabs(y-y_min), fabs(y-y_max))) +
                        sqr(min(fabs(z-z_min), fabs(z-z_max))) );

            float t;
            int w;

            unpack_tsdf(*pos,t,w);
            pack_tsdf (t, (int)round(w/(d/20.+1.)), *pos);
        }


  }
  }
}




void
pcl::device::initVolume (PtrStep<short2> volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

  initializeVolume<<<grid, block>>>(volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

//SEMA
void
pcl::device::cutVolume (PtrStep<short2> volume, pcl::ModelCoefficients::Ptr box_boundaries,
                        pcl::ModelCoefficients::Ptr plane_coeffs)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

//  std::cout<<"my plane: "<< plane_coeffs->values[0]<<std::endl;

  cut_Volume<<<grid, block>>>(volume, box_boundaries->values[0], box_boundaries->values[1],
                               box_boundaries->values[2], box_boundaries->values[3],
                              box_boundaries->values[4], box_boundaries->values[5],
                              plane_coeffs->values[0], plane_coeffs->values[1],
                              plane_coeffs->values[2], plane_coeffs->values[3]);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

//SEMA
void
pcl::device::reduceVolumeWeights (PtrStep<short2> volume, pcl::ModelCoefficients::Ptr box_boundaries)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

  reduce_Volume_Weights<<<grid, block>>>(volume, box_boundaries->values[0], box_boundaries->values[1],
                               box_boundaries->values[2], box_boundaries->values[3],
                              box_boundaries->values[4], box_boundaries->values[5]);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}
namespace pcl
{
  namespace device
  {
    struct Tsdf
    {
      enum
      {
        CTA_SIZE_X = 32, CTA_SIZE_Y = 8,
        MAX_WEIGHT = 1 << 7 //1 << 7   //DONT FORGET TO CHANGE RAYCASTER MAXWEIGHT!!!!!!!
      };

      mutable PtrStep<short2> volume;
      float3 cell_size;

      Intr intr;

      Mat33 Rcurr_inv;
      float3 tcurr;

      PtrStepSz<ushort> depth_raw; //depth in mm

      float tranc_dist_mm;

      __device__ __forceinline__ float3
      getVoxelGCoo (int x, int y, int z) const
      {
        float3 coo = make_float3 (x, y, z);
        coo += 0.5f;         //shift to cell center;

        coo.x *= cell_size.x;
        coo.y *= cell_size.y;
        coo.z *= cell_size.z;

        return coo;
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

        short2 *pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(*pos);

        for (int z = 0; z < VOLUME_Z; ++z, pos += elem_step)
        {
          float3 v_g = getVoxelGCoo (x, y, z);            //3 // p

          //tranform to curr cam coo space
          float3 v = Rcurr_inv * (v_g - tcurr);           //4

          int2 coo;           //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)           //6
          {
            int Dp = depth_raw.ptr (coo.y)[coo.x];

            if (Dp != 0)
            {
              float xl = (coo.x - intr.cx) / intr.fx;
              float yl = (coo.y - intr.cy) / intr.fy;
              float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

              float sdf = 1000 * norm (tcurr - v_g) * lambda_inv - Dp; //mm

              sdf *= (-1);

              if (sdf >= -tranc_dist_mm)
              {
                float tsdf = fmin (1, sdf / tranc_dist_mm);

                int weight_prev;
                float tsdf_prev;

                //read and unpack
                unpack_tsdf (*pos, tsdf_prev, weight_prev);

                const int Wrk = 1;

                float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                int weight_new = min (weight_prev + Wrk, MAX_WEIGHT);

                pack_tsdf (tsdf_new, weight_new, *pos);
              }
            }
          }
        }
      }
    };

    __global__ void
    integrateTsdfKernel (const Tsdf tsdf) {
      tsdf ();
    }

    __global__ void
    tsdf2 (PtrStep<short2> volume, const float tranc_dist_mm, const Mat33 Rcurr_inv, float3 tcurr,
           const Intr intr, const PtrStepSz<ushort> depth_raw, const float3 cell_size)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;

      short2 *pos = volume.ptr (y) + x;
      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

      float v_x = Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z;
      float v_y = Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z;
      float v_z = Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z;

//#pragma unroll
      for (int z = 0; z < VOLUME_Z; ++z)
      {
        float3 vr;
        vr.x = v_g_x;
        vr.y = v_g_y;
        vr.z = (v_g_z + z * cell_size.z);

        float3 v;
        v.x = v_x + Rcurr_inv.data[0].z * z * cell_size.z;
        v.y = v_y + Rcurr_inv.data[1].z * z * cell_size.z;
        v.z = v_z + Rcurr_inv.data[2].z * z * cell_size.z;

        int2 coo;         //project to current cam
        coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
        coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);


        if (v.z > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < depth_raw.cols && coo.y < depth_raw.rows)         //6
        {
          int Dp = depth_raw.ptr (coo.y)[coo.x]; //mm

          if (Dp != 0)
          {
            float xl = (coo.x - intr.cx) / intr.fx;
            float yl = (coo.y - intr.cy) / intr.fy;
            float lambda_inv = rsqrtf (xl * xl + yl * yl + 1);

            float sdf = Dp - norm (vr) * lambda_inv * 1000; //mm


            if (sdf >= -tranc_dist_mm)
            {
              float tsdf = fmin (1.f, sdf / tranc_dist_mm);

              int weight_prev;
              float tsdf_prev;

              //read and unpack
              unpack_tsdf (*pos, tsdf_prev, weight_prev);

              const int Wrk = 1;

              float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
              int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

              pack_tsdf (tsdf_new, weight_new, *pos);
            }
          }
        }
        pos += elem_step;
      }       /* for(int z = 0; z < VOLUME_Z; ++z) */
    }      /* __global__ */
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth_raw, const Intr& intr, const float3& volume_size,
                                  const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist,
                                  PtrStep<short2> volume)
{
  Tsdf tsdf;

  tsdf.volume = volume;
  tsdf.cell_size.x = volume_size.x / VOLUME_X;
  tsdf.cell_size.y = volume_size.y / VOLUME_Y;
  tsdf.cell_size.z = volume_size.z / VOLUME_Z;

  tsdf.intr = intr;

  tsdf.Rcurr_inv = Rcurr_inv;
  tsdf.tcurr = tcurr;
  tsdf.depth_raw = depth_raw;

  tsdf.tranc_dist_mm = tranc_dist*1000; //mm

  dim3 block (Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

#if 0
   //tsdf2<<<grid, block>>>(volume, tranc_dist, Rcurr_inv, tcurr, intr, depth_raw, tsdf.cell_size);
   integrateTsdfKernel<<<grid, block>>>(tsdf);
#endif
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}


namespace pcl
{
  namespace device
  {
    __global__ void
    scaleDepth (const PtrStepSz<ushort> depth, PtrStep<float> scaled, const Intr intr, PtrStep<short2> volume, const Mat33 Rcurr_, const float3 tcurr, const float3 cell_size)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= depth.cols || y >= depth.rows)
        return;

      int Dp = depth.ptr (y)[x];

      float xl = (x - intr.cx) / intr.fx;
      float yl = (y - intr.cy) / intr.fy;
      float lambda = sqrtf (xl * xl + yl * yl + 1);

      scaled.ptr (y)[x] = Dp * lambda/1000.f; //meters


//      //relocate sample point to avoid creating holes on the surface
//      float xn = Rcurr_.data[0].x * xl + Rcurr_.data[0].y * yl + Rcurr_.data[0].z ;
//      float yn = Rcurr_.data[1].x * xl + Rcurr_.data[1].y * yl + Rcurr_.data[1].z ;
//      float zn = Rcurr_.data[2].x * xl + Rcurr_.data[2].y * yl + Rcurr_.data[2].z ;

//      float norm = sqrtf(xn*xn+yn*yn + zn*zn);
//      xn /= norm;
//      yn /= norm;
//      zn /= norm;

//      int elem_step = volume.step * VOLUME_Y / sizeof(short2);
//      bool neg_before_sample = false;

//      for(float d=-0.01; d<=0.003 ; d+=0.001)
//      {
//          float dp = scaled.ptr (y)[x] + d;

//          int xx = round((dp*xn + tcurr.x)/cell_size.x);
//          int yy = round((dp*yn + tcurr.y)/cell_size.y);
//          int zz = round((dp*zn + tcurr.z)/cell_size.z);

//          if(xx>=0 && yy>=0 && zz>=0 && xx<VOLUME_X && yy<VOLUME_Y && zz<VOLUME_Z)
//          {
//              float tsdf_neighbor;
//              int weight_neighbor;
//              short2* pos2 = volume.ptr (yy) + xx + (zz)*elem_step;
//              unpack_tsdf (*pos2, tsdf_neighbor, weight_neighbor);


//              if(neg_before_sample && tsdf_neighbor>0.f)
//              {
//                  if(d<0)
//                      scaled.ptr (y)[x] = (dp - 0.003);
//                  else
//                      scaled.ptr (y)[x] -= 0.003;

////                  printf("%f depth from %f to %f\n", d, a, scaled.ptr (y)[x]);
//                  return;
//              }

//              if(tsdf_neighbor<0.f)
//                  neg_before_sample = true;
//              else
//                  neg_before_sample = false;

//          }
//      }
    }



    __global__ void
    tsdf23 (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
            const float tranc_dist, const Mat33 Rcurr_inv, const Mat33 Rcurr_, const float3 tcurr, const Intr intr, const float3 cell_size,
            float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, bool roi_selected, PtrStep<float> nmap)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= VOLUME_X || y >= VOLUME_Y)
        return;


      //SEMA - don't integrate completed parts!
      bool is_inside = false;
      if (roi_selected && x > x_min && x < x_max && y > y_min && y < y_max )
        is_inside = true;

      //SEMA
      //max distance from the cube
      float D = 40;
      const float qnan = numeric_limits<float>::quiet_NaN();

//SEMA
#define PI 22.0/7.0
#define cube(a) a*a*a
#define H 0.001
//#define WeightFunction(d) max((0.001/(64.*PI*cube(H)))*cube(1.-(d*d)/(H*H)), 0.)
#define WeightFunction(d) max(cube(1.-(d*d)/(H*H)), 0.)


      float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
      float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
      float v_g_z = (0. + 0.5f) * cell_size.z - tcurr.z;

      float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

      float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
      float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
      float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

      float z_scaled = 0;

      float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
      float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

      float tranc_dist_inv = 1.0f / tranc_dist;

      short2* pos = volume.ptr (y) + x;

      int elem_step = volume.step * VOLUME_Y / sizeof(short2);

//#pragma unroll
      for (int z = 0; z < VOLUME_Z;
           ++z,
           v_g_z += cell_size.z,
           z_scaled += cell_size.z,
           v_x += Rcurr_inv_0_z_scaled,
           v_y += Rcurr_inv_1_z_scaled,
           pos += elem_step)
      {
          float d = D;
          if(roi_selected){
              //SEMA
              if (is_inside && z > z_min && z < z_max)
                continue;

              //SEMA
              //distance to the ROI

              if (is_inside)
                  d = min(fabs(z-z_min), fabs(z-z_max));
              else if(z > z_min && z < z_max && y > y_min && y < y_max)
                  d = min(fabs(x-x_min), fabs(x-x_max));
              else if(z > z_min && z < z_max && x > x_min && x < x_max)
                  d = min(fabs(y-y_min), fabs(y-y_max));
              else if(z > z_min && z < z_max)
                  d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) +
                          sqr(min(fabs(y-y_min), fabs(y-y_max))) );
              else if(x > x_min && x < x_max)
                  d = sqrt( sqr(min(fabs(z-z_min), fabs(z-z_max))) +
                          sqr(min(fabs(y-y_min), fabs(y-y_max))) );
              else if(y > y_min && y < y_max)
                  d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) +
                          sqr(min(fabs(z-z_min), fabs(z-z_max))) );
              else
                  d = sqrt( sqr(min(fabs(x-x_min), fabs(x-x_max))) + sqr(min(fabs(y-y_min), fabs(y-y_max))) +
                          sqr(min(fabs(z-z_min), fabs(z-z_max))) );
          }

        float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
        if (inv_z < 0)
            continue;

        // project to current cam
        int2 coo =
        {
          __float2int_rn (v_x * inv_z + intr.cx),
          __float2int_rn (v_y * inv_z + intr.cy)
        };

        if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
        {
          float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters
//float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

//          //sema
//          float3 normal;
//          normal.x = nmap.ptr (coo.y      )[coo.x];
//          normal.y = nmap.ptr (coo.y + depthScaled.rows)[coo.x];
//          normal.z = nmap.ptr (coo.y + 2 * depthScaled.rows)[coo.x];

//          //sema
//          int Dp = depthScaled.ptr (coo.y)[coo.x];
//          float3 ray;
//          ray.x = (coo.x - intr.cx)/intr.fx;
//          ray.x = (coo.y - intr.cy)/intr.fy;
//          ray.z = Dp;
////          Dp /= sqrtf(ray.x*ray.x+ray.y*ray.y+1);
////          ray.x *= Dp;
////          ray.y *= Dp;
////          ray.z = Dp;

//          float3 vox_;
//          vox_.x = v_g_x;
//          vox_.y = v_g_y;
//          vox_.z = v_g_z;
//          float theta = dot(ray, vox_)/sqrtf(dot(ray,ray)*dot(vox_,vox_));



          float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);



//          float w_d = sdf;
//          float nt = 1.;
//          if (/*theta > 0.99 &&*/ normal.x != qnan && normal.y != qnan && normal.z != qnan)
//          {
//              float norm2 = dot(normal, normal);
//              if (norm2 >= 1e-6 && norm2 < 10.)
//              {
//                  normal *= rsqrt(norm2);


////                  w_d = theta;

//                  nt = fabs(v_g_x * normal.x + v_g_y * normal.y + v_g_z * normal.z);

//                  nt *= rsqrt(v_g_x * v_g_x + v_g_y * v_g_y + v_g_z * v_g_z);

////                  w_d = sqrtf(sdf*sdf*(1. - nt*nt));
////                  sdf*=nt;
////                  sdf = sdf/fabs(sdf)*fabs((ray.x-vox_.x) * normal.x + (ray.y-vox_.y) * normal.y + (ray.z-vox_.z) * normal.z);

////                  if (nt < 0.5)
////                      integrate = false;
//              }
//          }



          if (Dp_scaled != 0   && sdf >=  -tranc_dist) //meters
              {


                  //              float nt = v_g_x * normal.x + v_g_y * normal.y + v_g_z * normal.z;
                  //              float cosine = nt ;//= nt * rsqrt(v_g_x * v_g_x + v_g_y * v_g_y + v_g_z * v_g_z);
                  //              if(cosine<0.1)
                  //                  continue;

                  float tsdf = fmin (1.0f, (sdf * tranc_dist_inv));
//                  if(sdf>0)
////                      tsdf =sdf * tranc_dist_inv;
////                  else
//                      tsdf =fmin(1.0f, sdf / (tranc_dist*2));
                  const int Wrk = round(D*min(d/D, 1.));

                  //read and unpack
                  float tsdf_prev;
                  int weight_prev;
                  unpack_tsdf (*pos, tsdf_prev, weight_prev);

////                  bool stop = false;

////                  float cri_dist = 0.00f;

//                  bool hit_surface_before_sample = false;
////                  if(tsdf > 0.0f && tsdf_prev > cri_dist && ((tsdf_prev * weight_prev + Wrk * tsdf)) <= cri_dist)
////                  {
////                      printf("\nsample point: %f %f %f\n", v_x/intr.fx, v_y/intr.fy, 1/inv_z);
////                      printf("\nsample point: %d %d %d\n", x, y, z);
//                      bool neg_before_sample = false;
//                      int dn = -3;
//                      int dp = 1;
////                      if(tsdf>0.f)
////                          dp += tsdf/cell_size.z;
//                      if(tsdf<=0.f)
//                          dn += tsdf/cell_size.z;
//                      for(int d=dn; d<=dp ; d++)
//                      {
//                          float z_ = 1./inv_z + d*cell_size.z;
//                          float x_ = (coo.x - intr.cx)*z_/intr.fx;
//                          float y_ = (coo.y - intr.cy)*z_/intr.fy;

////                          printf("ray point %d: %f %f %f\n", d, x_, y_, z_);



//                          float xn = Rcurr_.data[0].x * x_ + Rcurr_.data[0].y * y_ + Rcurr_.data[0].z * z_;
//                          float yn = Rcurr_.data[1].x * x_ + Rcurr_.data[1].y * y_ + Rcurr_.data[1].z * z_;
//                          float zn = Rcurr_.data[2].x * x_ + Rcurr_.data[2].y * y_ + Rcurr_.data[2].z * z_;

//                          int xx = round((xn + tcurr.x)/cell_size.x);
//                          int yy = round((yn + tcurr.y)/cell_size.y);
//                          int zz = round((zn + tcurr.z)/cell_size.z);

////                           printf("ray point %d: %d %d %d - %d %d %d \n", d, x, y, z, xx,yy,zz);

//                          if(xx>=0 && yy>=0 && zz>=0 && xx<VOLUME_X && yy<VOLUME_Y && zz<VOLUME_Z)
//                          {
//                              float tsdf_neighbor;
//                              int weight_neighbor;
//                              short2* pos2 = volume.ptr (yy) + xx + (zz)*elem_step;
//                              unpack_tsdf (*pos2, tsdf_neighbor, weight_neighbor);

//                              if(tsdf_neighbor<0.f)
//                                  neg_before_sample = true;
//                              else if(neg_before_sample && tsdf_neighbor>=0.f)
//                              {
//                                  hit_surface_before_sample = true;
//                                  if(d>0)
//                                      tsdf = sqrt( sqr((xx-x)*cell_size.x) + sqr((yy-y)*cell_size.y) + sqr((zz-z)*cell_size.z)) + 0.0001;
//                                  else
//                                    tsdf = 0.0f;
//                                  break;
//                              }
//                          }
//                      }
//                  }
//                  if(!hit_surface_before_sample)
                  {

                      float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                      int weight_new = min (weight_prev + Wrk, (int)D*Tsdf::MAX_WEIGHT);

                      pack_tsdf (tsdf_new, weight_new, *pos);
                  }

//                  if(tsdf_prev < 0.f && ((tsdf_prev * weight_prev + Wrk * tsdf)) > 0.f)
//                  {
//                      stop = true;
//                  }
//                  if(tsdf_prev > 0.f && ((tsdf_prev * weight_prev + Wrk * tsdf)) <= 0.f)
//                  {
//                      stop = true;
//                  }
//                  if(tsdf_prev < 0.f && ((tsdf_prev * weight_prev + Wrk * tsdf)) > 0.f)
//                  {
//                      float tsdf_neighbor;
//                      int weight_neighbor;
//                      for(int dx=max(-1, -x); dx<=min(1, VOLUME_X-x); dx++)
//                      {
//                          for(int dy=max(-1, -y); dy<=min(1, VOLUME_Y-y); dy++)
//                          {
//                              for(int dz=max(-1, -z); dz<=min(1, VOLUME_Z-z); dz++)
//                              {
//                                  if(dx == 0 && dy == 0 && dz ==0)    continue;

//                                  short2* pos2 = volume.ptr (y+dy) + x+dx + (z+dz)*elem_step;
//                                  unpack_tsdf (*pos2, tsdf_neighbor, weight_neighbor);
//                                  if(tsdf_neighbor > 0.f)
//                                      stop = true;
//                              }
//                          }
//                      }

////                      float z_ = 1./inv_z + cell_size.z;
////                      float x_ = (coo.x - intr.cx)*z_/intr.fx;
////                      float y_ = (coo.y - intr.cy)*z_/intr.fy;


////                      float xn = Rcurr_.data[0].x * x_ + Rcurr_.data[0].y * y_ + Rcurr_.data[0].z * z_;
////                      float yn = Rcurr_.data[1].x * x_ + Rcurr_.data[1].y * y_ + Rcurr_.data[1].z * z_;
////                      float zn = Rcurr_.data[2].x * x_ + Rcurr_.data[2].y * y_ + Rcurr_.data[2].z * z_;

////                      int xx = round((xn + tcurr.x)/cell_size.x);
////                      int yy = round((yn + tcurr.y)/cell_size.y);
////                      int zz = round((zn + tcurr.z)/cell_size.z);

////                      if(xx>0 && yy>0 && zz>0 && xx<=VOLUME_X && yy<=VOLUME_Y && zz<=VOLUME_Z)
////                      {
////                          short2* pos2 = volume.ptr (yy) + xx + (zz)*elem_step;
////                          unpack_tsdf (*pos2, tsdf_neighbor, weight_neighbor);
////                          if(tsdf_neighbor > 0.f)
////                              stop = true;
////                      }
//                  }
//                  if(!stop)
//                  {
//                      //            const int Wrk = 1;
//                      //SEMA
//                      //linear weighting
//    //                  if(sdf < 0.008 && tsdf_prev < 0. && tsdf_prev > -0.5 )
//    //                      continue;

//                      /*const int Wrk = round(D*min(d/D, 1.));*///round(max((3./4.*D)*WeightFunction(sdf),0.) + 0.25*D);//round(D*min(d/D, 1.));//round((float)D*min(d/D, 1.)/(WeightFunction(w_d)+1.) );//round((float)D*(max(10.*w_d-8.,0.1)) );//round((float)D*(WeightFunction(w_d)) );//round(D*min(d/D, 1.));

//                      float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
//                      int weight_new = min (weight_prev + Wrk, (int)D*Tsdf::MAX_WEIGHT);

//                      pack_tsdf (tsdf_new, weight_new, *pos);
//                  }

              }
//          if(Dp_scaled == 0)
//          {
//              float tsdf = 1.;
////                   float tsdf = sdf ;//* tranc_dist_inv;

//              //read and unpack
//              float tsdf_prev;
//              int weight_prev;
//              unpack_tsdf (*pos, tsdf_prev, weight_prev);

//              //            const int Wrk = 1;
//              //SEMA
//              //linear weighting
//              const int Wrk = D;//round((float)D*min(d/D, 1.)/(WeightFunction(w_d)+1.) );//round((float)D*(max(10.*w_d-8.,0.1)) );//round((float)D*(WeightFunction(w_d)) );//round(D*min(d/D, 1.));

//              float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
//              int weight_new = min (weight_prev + Wrk, (int)D*Tsdf::MAX_WEIGHT);

//              pack_tsdf (tsdf_new, weight_new, *pos);
//          }


        }
      }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__

    __global__ void
    tsdf23normal_hack (const PtrStepSz<float> depthScaled, PtrStep<short2> volume,
                  const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
            return;

        const float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
        const float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
        float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

        float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

        float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
        float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
        float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

        float z_scaled = 0;

        float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
        float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

        float tranc_dist_inv = 1.0f / tranc_dist;

        short2* pos = volume.ptr (y) + x;
        int elem_step = volume.step * VOLUME_Y / sizeof(short2);

        //#pragma unroll
        for (int z = 0; z < VOLUME_Z;
            ++z,
            v_g_z += cell_size.z,
            z_scaled += cell_size.z,
            v_x += Rcurr_inv_0_z_scaled,
            v_y += Rcurr_inv_1_z_scaled,
            pos += elem_step)
        {
            float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
            if (inv_z < 0)
                continue;

            // project to current cam
            int2 coo =
            {
                __float2int_rn (v_x * inv_z + intr.cx),
                __float2int_rn (v_y * inv_z + intr.cy)
            };

            if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
            {
                float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

                float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

                if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
                {
                    float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

                    float cosine = 1.;

                    bool integrate = true;
                    if ((x > 0 &&  x < VOLUME_X-2) && (y > 0 && y < VOLUME_Y-2) && (z > 0 && z < VOLUME_Z-2))
                    {
                        const float qnan = numeric_limits<float>::quiet_NaN();
                        float3 normal = make_float3(qnan, qnan, qnan);

                        float Fn, Fp;
                        int Wn = 0, Wp = 0;
                        unpack_tsdf (*(pos + elem_step), Fn, Wn);
                        unpack_tsdf (*(pos - elem_step), Fp, Wp);

                        if (Wn > 16 && Wp > 16)
                            normal.z = (Fn - Fp)/cell_size.z;

                        unpack_tsdf (*(pos + volume.step/sizeof(short2) ), Fn, Wn);
                        unpack_tsdf (*(pos - volume.step/sizeof(short2) ), Fp, Wp);

                        if (Wn > 16 && Wp > 16)
                            normal.y = (Fn - Fp)/cell_size.y;

                        unpack_tsdf (*(pos + 1), Fn, Wn);
                        unpack_tsdf (*(pos - 1), Fp, Wp);

                        if (Wn > 16 && Wp > 16)
                            normal.x = (Fn - Fp)/cell_size.x;

                        if (normal.x != qnan && normal.y != qnan && normal.z != qnan)
                        {
                            float norm2 = dot(normal, normal);
                            if (norm2 >= 1e-10)
                            {
                                normal *= rsqrt(norm2);

                                float nt = v_g_x * normal.x + v_g_y * normal.y + v_g_z * normal.z;
                                cosine = nt * rsqrt(v_g_x * v_g_x + v_g_y * v_g_y + v_g_z * v_g_z);

                                if (cosine < 0.5)
                                    integrate = false;
                            }
                        }
                    }

                    int D=40;

                    if (integrate)
                    {
                        //read and unpack
                        float tsdf_prev;
                        int weight_prev;
                        unpack_tsdf (*pos, tsdf_prev, weight_prev);

                        const int Wrk = (int)round(40.*cosine);//1;

                        float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                        int weight_new = min (weight_prev + Wrk, (int)D*Tsdf::MAX_WEIGHT);

                        pack_tsdf (tsdf_new, weight_new, *pos);
                    }
                }
            }
        }       // for(int z = 0; z < VOLUME_Z; ++z)
    }      // __global__
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateTsdfVolume (const PtrStepSz<ushort>& depth, const Intr& intr,
                                  const float3& volume_size, const Mat33& Rcurr_inv, const Mat33& Rcurr_, const float3& tcurr,
                                  float tranc_dist,
                                  PtrStep<short2> volume, DeviceArray2D<float>& depthScaled,
                                  pcl::ModelCoefficients::Ptr box_boundaries, bool roi_selected, MapArr& nmap)
{
    float3 cell_size;
    cell_size.x = volume_size.x / VOLUME_X;
    cell_size.y = volume_size.y / VOLUME_Y;
    cell_size.z = volume_size.z / VOLUME_Z;

  depthScaled.create (depth.rows, depth.cols);

  dim3 block_scale (32, 8);
  dim3 grid_scale (divUp (depth.cols, block_scale.x), divUp (depth.rows, block_scale.y));

  //scales depth along ray and converts mm -> meters.
  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr, volume, Rcurr_, tcurr, cell_size);
  cudaSafeCall ( cudaGetLastError () );


  //dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
  dim3 block (16, 16);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

  tsdf23<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, Rcurr_, tcurr, intr, cell_size,
                          box_boundaries->values[0], box_boundaries->values[1],
                          box_boundaries->values[2], box_boundaries->values[3],
                          box_boundaries->values[4], box_boundaries->values[5], roi_selected, nmap);
//  tsdf23normal_hack<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}
