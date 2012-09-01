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

#include "pcl/gpu/utils/device/vector_math.hpp"

namespace pcl
{
  namespace device
  {
    __global__ void
    initColorVolumeKernel (PtrStep<uchar4> volume)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < VOLUME_X && y < VOLUME_Y)
      {
        uchar4 *pos = volume.ptr (y) + x;
        int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
        for (int z = 0; z < VOLUME_Z; ++z, pos += z_step)
          *pos = make_uchar4 (0, 0, 0, 0);
      }
    }
  }
}

void
pcl::device::initColorVolume (PtrStep<uchar4> color_volume)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp (VOLUME_X, block.x);
  grid.y = divUp (VOLUME_Y, block.y);

  initColorVolumeKernel<<<grid, block>>>(color_volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    struct ColorVolumeImpl
    {
      enum
      {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,

        ONE_VOXEL = 0
      };

      Intr intr;

      PtrStep<float> vmap;
      PtrStepSz<uchar3> colors;

      Mat33 R_inv;
      float3 t;

      float3 cell_size;
      float tranc_dist;

      int max_weight;

      mutable PtrStep<uchar4> color_volume;

      __device__ __forceinline__ int3
      getVoxel (float3 point) const
      {
        int vx = __float2int_rd (point.x / cell_size.x);                // round to negative infinity
        int vy = __float2int_rd (point.y / cell_size.y);
        int vz = __float2int_rd (point.z / cell_size.z);

        return make_int3 (vx, vy, vz);
      }

      __device__ __forceinline__ float3
      getVoxelGCoo (int x, int y, int z) const
      {
        float3 coo = make_float3 (x, y, z);
        coo += 0.5f;                 //shift to cell center;

        coo.x *= cell_size.x;
        coo.y *= cell_size.y;
        coo.z *= cell_size.z;

        return coo;
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= VOLUME_X || y >= VOLUME_Y)
          return;

        for (int z = 0; z < VOLUME_X; ++z)
        {
          float3 v_g = getVoxelGCoo (x, y, z);

          float3 v = R_inv * (v_g - t);

          if (v.z <= 0)
            continue;

          int2 coo;                   //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (coo.x >= 0 && coo.y >= 0 && coo.x < colors.cols && coo.y < colors.rows)
          {
            float3 p;
            p.x = vmap.ptr (coo.y)[coo.x];

            if (isnan (p.x))
              continue;

            p.y = vmap.ptr (coo.y + colors.rows    )[coo.x];
            p.z = vmap.ptr (coo.y + colors.rows * 2)[coo.x];

            bool update = false;
            if (ONE_VOXEL)
            {
              int3 vp = getVoxel (p);
              update = vp.x == x && vp.y == y && vp.z == z;
            }
            else
            {
              float dist = norm (p - v_g);
              update = dist < tranc_dist;
            }

            if (update)
            {
              uchar4 *ptr = color_volume.ptr (VOLUME_Y * z + y) + x;
              uchar3 rgb = colors.ptr (coo.y)[coo.x];
              uchar4 volume_rgbw = *ptr;

              int weight_prev = volume_rgbw.w;

              const float Wrk = 1.f;
              float new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
              float new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
              float new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

              int weight_new = weight_prev + 1;

              uchar4 volume_rgbw_new;
              volume_rgbw_new.x = min (255, max (0, __float2int_rn (new_x)));
              volume_rgbw_new.y = min (255, max (0, __float2int_rn (new_y)));
              volume_rgbw_new.z = min (255, max (0, __float2int_rn (new_z)));
              volume_rgbw_new.w = min (max_weight, weight_new);

              *ptr = volume_rgbw_new;
            }
          }           /* in camera image range */
        }         /* for(int z = 0; z < VOLUME_X; ++z) */
      }       /* void operator() */
    };

    __global__ void
    updateColorVolumeKernel (const ColorVolumeImpl cvi) {
      cvi ();
    }
  }
}

void
pcl::device::updateColorVolume (const Intr& intr, float tranc_dist, const Mat33& R_inv, const float3& t,
                                const MapArr& vmap, const PtrStepSz<uchar3>& colors, const float3& volume_size, PtrStep<uchar4> color_volume, int max_weight)
{
  ColorVolumeImpl cvi;
  cvi.vmap = vmap;
  cvi.colors = colors;
  cvi.color_volume = color_volume;

  cvi.R_inv = R_inv;
  cvi.t = t;
  cvi.intr = intr;
  cvi.tranc_dist = tranc_dist;
  cvi.max_weight = min (max (0, max_weight), 255);

  cvi.cell_size.x = volume_size.x / VOLUME_X;
  cvi.cell_size.y = volume_size.y / VOLUME_Y;
  cvi.cell_size.z = volume_size.z / VOLUME_Z;

  dim3 block (ColorVolumeImpl::CTA_SIZE_X, ColorVolumeImpl::CTA_SIZE_Y);
  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

  updateColorVolumeKernel<<<grid, block>>>(cvi);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    __global__ void
    extractColorsKernel (const float3 cell_size, const PtrStep<uchar4> color_volume, const PtrSz<PointType> points, uchar4 *colors)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx < points.size)
      {
        int3 v;
        float3 p = *(const float3*)(points.data + idx);
        v.x = __float2int_rd (p.x / cell_size.x);        // round to negative infinity
        v.y = __float2int_rd (p.y / cell_size.y);
        v.z = __float2int_rd (p.z / cell_size.z);

        uchar4 rgbw = color_volume.ptr (VOLUME_Y * v.z + v.y)[v.x];
        colors[idx] = make_uchar4 (rgbw.z, rgbw.y, rgbw.x, 0); //bgra
      }
    }
  }
}

void
pcl::device::exctractColors (const PtrStep<uchar4>& color_volume, const float3& volume_size, const PtrSz<PointType>& points, uchar4* colors)
{
  const int block = 256;
  float3 cell_size = make_float3 (volume_size.x / VOLUME_X, volume_size.y / VOLUME_Y, volume_size.z / VOLUME_Z);
  extractColorsKernel<<<divUp (points.size, block), block>>>(cell_size, color_volume, points, colors);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
};

///////////////////////////////////////////////////////
//SEMA
//average colors from rgb images to assign 3d model color values
//////////////////////////////////////////////////////


namespace pcl
{
  namespace device
  {

      __global__ void
      setBlackKernel (uchar4* color_cloud, int size)
      {
          int x = threadIdx.x + blockIdx.x * blockDim.x;
          if(x > size)
              return;
          color_cloud[x] = make_uchar4(0,0,0,0);
      }
      __global__ void
      image23 (PtrStepSz<uchar3> rgb, PtrSz<PointType> cloud, uchar4* color_cloud, PtrStep<float> vmap, PtrStep<float> nmap,
               const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size)
      {
          int x = threadIdx.x + blockIdx.x * blockDim.x;

          if(x > cloud.size)
              return;


          float3 v_g = *(float3*)(cloud.data + x);

          float3 v = Rcurr_inv * (v_g - tcurr);

          if (v.z <= 0)
              return;

          int2 coo;                   //project to current cam
          coo.x = __float2int_rn (v.x * intr.fx / v.z + intr.cx);
          coo.y = __float2int_rn (v.y * intr.fy / v.z + intr.cy);

          if (coo.x >= 0 && coo.y >= 0 && coo.x < rgb.cols && coo.y < rgb.rows)
          {
//              float N_Z_THRESHOLD = 0.2;

              float3 p;
              p.x = vmap.ptr (coo.y)[coo.x];

              if (isnan (p.x))
                  return;

              float n_z = nmap.ptr (coo.y + rgb.rows * 2)[coo.x];

              p.y = vmap.ptr (coo.y + rgb.rows    )[coo.x];
              p.z = vmap.ptr (coo.y + rgb.rows * 2)[coo.x];

              float dist = norm (v)  - norm(p);

              if( dist > -0.002 && dist < 0.008)
              {
                  uchar4 rgbw = color_cloud[x];
                  uchar3 pix = rgb.ptr (coo.y)[coo.x];

//                  if(rgbw.w > 235)
//                      return;

//                  float wn = 20.f*n_z;

//                  if(rgbw.w < wn)
//                  {
//                      rgbw = make_uchar4( min (255, max (0, pix.z) ),
//                                          min (255, max (0, pix.y) ),
//                                          min (255, max (0, pix.x) ),
//                                          min (255, __float2int_rn (wn) ));//bgra

//                  }
//                  else if(wn >= N_Z_THRESHOLD) //blend
//                  {
//                      rgbw = make_uchar4( min (255, max (0, __float2int_rn ((float)(rgbw.z*rgbw.w + wn*pix.z)/((float)rgbw.w+wn)) )),
//                                          min (255, max (0, __float2int_rn ((float)(rgbw.y*rgbw.w + wn*pix.y)/((float)rgbw.w+wn)))),
//                                          min (255, max (0, __float2int_rn ((float)(rgbw.x*rgbw.w + wn*pix.x)/((float)rgbw.w+wn)))),
//                                          min (255, __float2int_rn (wn + rgbw.w) ) );//bgra

////                      rgbw = make_uchar4( min (255, max (0, __float2int_rn ((float)(rgbw.z*rgbw.w + pix.z)/((float)rgbw.w+1.)))) ,
////                                          min (255, max (0, __float2int_rn ((float)(rgbw.y*rgbw.w + pix.y)/((float)rgbw.w+1.)))) ,
////                                          min (255, max (0, __float2int_rn ((float)(rgbw.x*rgbw.w + pix.x)/((float)rgbw.w+1.)))) ,
////                                          min (255, rgbw.w+1) );//bgra
//                  }
//                  else
//                  {
//                      rgbw = make_uchar4( min (255, max (0, pix.z) ),
//                                          min (255, max (0, pix.y) ),
//                                          min (255, max (0, pix.x) ),
//                                          min (255, __float2int_rn (wn) ));//bgra
//                  }

                  //Average
//                  if(n_z > N_Z_THRESHOLD)
                      rgbw = make_uchar4( min (255, max (0, __float2int_rn ((float)(rgbw.z*rgbw.w + pix.z)/((float)rgbw.w+1.)))) ,
                                          min (255, max (0, __float2int_rn ((float)(rgbw.y*rgbw.w + pix.y)/((float)rgbw.w+1.)))) ,
                                          min (255, max (0, __float2int_rn ((float)(rgbw.x*rgbw.w + pix.x)/((float)rgbw.w+1.)))) ,
                                          min (255, rgbw.w+1) );//bgra

                  color_cloud[x] = rgbw;
              }

          }
      }      // __global__
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::integrateColorFromRGB (PtrStepSz<uchar3> rgb, const Intr& intr,
                                  const float3& volume_size, const Mat33& Rcurr_inv, const float3& tcurr,
                                  float tranc_dist,
                                  const MapArr& vmap, const MapArr& nmap, PtrSz<PointType> cloud, uchar4* color_cloud, size_t sz)
{
//  depthScaled.create (rgb.rows, rgb.cols);

//  dim3 block_scale (32, 8);
//  dim3 grid_scale (divUp (rgb.cols, block_scale.x), divUp (rgb.rows, block_scale.y));

//  //scales depth along ray and converts mm -> meters.
//  scaleDepth<<<grid_scale, block_scale>>>(depth, depthScaled, intr);
//  cudaSafeCall ( cudaGetLastError () );

    const int block = 256;
    float3 cell_size = make_float3 (volume_size.x / VOLUME_X, volume_size.y / VOLUME_Y, volume_size.z / VOLUME_Z);
    image23<<<divUp (sz, block), block>>>(rgb, cloud, color_cloud, vmap, nmap, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());


//  float3 cell_size;
//  cell_size.x = volume_size.x / VOLUME_X;
//  cell_size.y = volume_size.y / VOLUME_Y;
//  cell_size.z = volume_size.z / VOLUME_Z;

//  dim3 block (16, 16);
////  dim3 grid (divUp (VOLUME_X, block.x), divUp (VOLUME_Y, block.y));

//  image23<<<n_blocks, block>>>(rgb, cloud, color_cloud, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

//  cudaSafeCall ( cudaGetLastError () );
//  cudaSafeCall (cudaDeviceSynchronize ());
};
void
pcl::device::setBlack ( uchar4* color_cloud, size_t sz)
{
    const int block = 256;
    setBlackKernel<<<divUp (sz, block), block>>>(color_cloud, sz);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());

};
