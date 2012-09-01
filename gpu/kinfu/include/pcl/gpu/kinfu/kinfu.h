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

#ifndef PCL_KINFU_KINFUTRACKER_HPP_
#define PCL_KINFU_KINFUTRACKER_HPP_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu/pixel_rgb.h>
#include <pcl/gpu/kinfu/tsdf_volume.h>
#include <pcl/gpu/kinfu/color_volume.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <vector>
#include <math.h>

/*SEMA
  */
#include <string.h>
#include <time.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/features/normal_3d.h>

#include <pcl/gpu/kinfu/video_recorder.h>

#include "internal.h"
#include <pcl/io/ply_io.h>


struct HandModel{
    double sqr(double a) { return a*a;}
    void reset() {
        num_samples = 0;
        cube_dim = 96;
        scale = (float)cube_dim/256.;
        mean_rgb[0] = 0., mean_rgb[1] = 0., mean_rgb[2] = 0.;
        std_dev[0] = 0., std_dev[1] = 0., std_dev[2] = 0.;
        samples.clear();
        isset = false;

        color_cube.clear();
        for(int i=0; i<cube_dim; ++i)
            color_cube.push_back(std::vector< std::vector< int > >());
        for(int i=0; i<cube_dim; ++i)
            for(int j=0; j<cube_dim; ++j)
                color_cube[i].push_back(std::vector< int >());
        for(int i=0; i<cube_dim; ++i)
            for(int j=0; j<cube_dim; ++j)
                for(int k=0; k<cube_dim; ++k)
                    color_cube[i][j].push_back(0);
    }
    void calculateModel() {
        num_samples = samples.size();
        printf("# samples: %d\n", num_samples);
        if(num_samples<100)
            return;

        for(int i=0; i<num_samples; ++i)
        {
            int u = round(((float)samples[i].r)*scale);
            int v = round(((float)samples[i].g)*scale);
            int z = round(((float)samples[i].b)*scale);

            u = std::max(0, std::min(u, cube_dim-1));
            v = std::max(0, std::min(v, cube_dim-1));
            z = std::max(0, std::min(z, cube_dim-1));

            if(u<cube_dim && v<cube_dim && z<cube_dim)
                color_cube[u][v][z] +=1;
            else
                printf("%d %d %d\n",u,v,z);
        }
        int threshold = 10;//round(num_samples/10000.);
        for(int i=0; i<cube_dim; ++i)
            for(int j=0; j<cube_dim; ++j)
                for(int k=0; k<cube_dim; ++k)
                    if(color_cube[i][j][k]<threshold)
                        color_cube[i][j][k] = 0;

        samples.clear();
        isset = true;
        printf("hand model is created!\n");
    }

    float mean_rgb[3];
    float std_dev[3];

    int cube_dim;
    float scale;
    std::vector< std::vector< std::vector< int > > > color_cube;
    std::vector< PixelRGB > samples;
    int num_samples;
    bool isset;
};

namespace pcl
{
  namespace gpu
  {        
    /** \brief KinfuTracker class encapsulates implementation of Microsoft Kinect Fusion algorithm
      * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
      */
    class PCL_EXPORTS KinfuTracker
    {
      public:
        /** \brief Pixel type for rendered image. */
        typedef pcl::gpu::PixelRGB PixelRGB;

        typedef DeviceArray2D<PixelRGB> View;
        typedef DeviceArray2D<unsigned short> DepthMap;

        typedef pcl::PointXYZ PointType;
        typedef pcl::Normal NormalType;

        /** \brief Constructor
          * \param[in] rows height of depth image
          * \param[in] cols width of depth image          
          */
        KinfuTracker (int rows = 480, int cols = 640);


        //SEMA
        // calculate base Plane(a, b, c, d)
        void getBasePlane ();
        bool init;
        ModelCoefficients::Ptr plane_coeffs;

        //SEMA
        //clean tsdf to extract object in ROI
        void extractObject(pcl::ModelCoefficients::Ptr box_boundaries,
                                             pcl::ModelCoefficients::Ptr coefficients);

        //SEMA
        void
        reduceTsdfWeights(pcl::ModelCoefficients::Ptr box_boundaries);

        /** \brief Sets Depth camera intrinsics
          * \param[in] fx focal length x 
          * \param[in] fy focal length y
          * \param[in] cx principal point x
          * \param[in] cy principal point y
          */
        void
        setDepthIntrinsics (float fx, float fy, float cx = -1, float cy = -1);

        pcl::device::Intr
        getDepthIntrinsics () {return device::Intr(fx_, fy_, cx_, cy_);}

        /** \brief Sets initial camera pose relative to volume coordiante space
          * \param[in] pose Initial camera pose
          */
        void
        setInitalCameraPose (const Eigen::Affine3f& pose);
                        
		/** \brief Sets truncation threshold for depth image for ICP step only! This helps 
		  *  to filter measurements that are outside tsdf volume. Pass zero to disable the truncation.
          * \param[in] max_icp_distance_ Maximal distance, higher values are reset to zero (means no measurement). 
          */
        void
        setDepthTruncationForICP (float max_icp_distance = 0.f);

        /** \brief Sets ICP filtering parameters.
          * \param[in] distThreshold distance.
          * \param[in] sineOfAngle sine of angle between normals.
          */
        void
        setIcpCorespFilteringParams (float distThreshold, float sineOfAngle);
        
        /** \brief Sets integration threshold. TSDF volume is integrated iff a camera movement metric exceedes the threshold value. 
          * The metric represents the following: M = (rodrigues(Rotation).norm() + alpha*translation.norm())/2, where alpha = 1.f (hardcoded constant)
          * \param[in] threshold a value to compare with the metric. Suitable values are ~0.001          
          */
        void
        setCameraMovementThreshold(float threshold = 0.001f);

        /** \brief Performs initialization for color integration. Must be called before calling color integration. 
          * \param[in] max_weight max weighe for color integration. -1 means default weight.
          */
        void
        initColorIntegration(int max_weight = -1);        

        /** \brief Returns cols passed to ctor */
        int
        cols ();

        /** \brief Returns rows passed to ctor */
        int
        rows ();

        /** \brief Processes next frame.
          * \param[in] Depth next frame with values in millimeters
          * \return true if can render 3D view.
          */
        bool operator() (const DepthMap& depth, View& view, bool integrate, bool roi_selected_,  View& rgb24);

        /** \brief Processes next frame (both depth and color integration). Please call initColorIntegration before invpoking this.
          * \param[in] depth next depth frame with values in millimeters
          * \param[in] colors next RGB frame
          * \return true if can render 3D view.
          */
        bool operator() (const DepthMap& depth, const View& colors, View& view, bool integrate, bool roi_selected_,  View& rgb24);

        /** \brief Returns camera pose at given time, default the last pose
          * \param[in] time Index of frame for which camera pose is returned.
          * \return camera pose
          */
        Eigen::Affine3f
        getCameraPose (int time = -1) const;

        //sema
        pcl::device::Mat33&
        getCameraRot (int time=-1);
        pcl::device::Mat33&
        getCameraRotInverse (int time=-1);
        float3&
        getCameraTrans (int time=-1) ;

        std::vector< pcl::device::Mat33 > r_inv_;
        std::vector< float3 > t_;

        /** \brief Returns number of poses including initial */
        size_t
        getNumberOfPoses () const;

        /** \brief Returns TSDF volume storage */
        const TsdfVolume& volume() const;

        /** \brief Returns TSDF volume storage */
        TsdfVolume& volume();

        /** \brief Returns color volume storage */
        const ColorVolume& colorVolume() const;

        /** \brief Returns color volume storage */
        ColorVolume& colorVolume();
        
        /** \brief Renders 3D scene to display to human
          * \param[out] view output array with image
          */
        void
        getImage (View& view);
        
        /** \brief Returns point cloud abserved from last camera pose
          * \param[out] cloud output array for points
          */
        void
        getLastFrameCloud (DeviceArray2D<PointType>& cloud) const;

        /** \brief Returns point cloud abserved from last camera pose
          * \param[out] normals output array for normals
          */
        void
        getLastFrameNormals (DeviceArray2D<NormalType>& normals) const;
///////////////////////////
        //SEMA
        /** \brief Returns point cloud observed from current frame
          * \param[out] cloud output array for points
          */
        void
        getCurrentFrameCloud (DeviceArray2D<PointType>& cloud) const;

        /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
          */
        void
        reset ();

        //SEMA
        void calculateImageRegistrationSIFT();
        void setROI(pcl::ModelCoefficients::Ptr box_boundaries);

        void raw_depth_to_bilateraled();

        bool reconstructWithModelProxy (PointCloud<PointXYZ>::Ptr cloud, int ctrl);
        bool reconstructWithModelProxy2 (PointCloud<PointXYZ>::Ptr cloud, int ctrl=1);

        bool reconstructWithModelProxy_NonLinearICP (PointCloud<PointXYZ>::Ptr cloud, int ctrl);

        //save time when roi is set
        int roi_selection_time;
        void set_ROI_selection_time() {roi_selection_time = global_time_;}

        // set the camera position to a previous stable position
        void stabilizeCameraPosition();
        int stable_global_time;

        TsdfVolume::Ptr tsdf_volume_;
        PointCloud<PointXYZ>::Ptr cloud_all;
        PointCloud<PointXYZ>::Ptr cloud_curr;

        ModelCoefficients::Ptr roi_boundaries_;
        std::vector<DeviceArray2D<float> > vmaps_;
        std::vector<DeviceArray2D<float> > nmaps_;

        std::vector<  std::vector<ushort> > depth_images;
        std::vector<  std::vector<ushort> > depth_raw_images;


        cv::Mat rgb_prev;
        cv::Mat rgb_curr;

        HandModel hand;
        bool ishand;
/////////////////////////////
      private:
        
        /** \brief Number of pyramid levels */
        enum { LEVELS = 1 };

        /** \brief ICP Correspondences  map type */
        typedef DeviceArray2D<int> CorespMap;

        /** \brief Vertex or Normal Map type */
        typedef DeviceArray2D<float> MapArr;
        
        typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
        typedef Eigen::Vector3f Vector3f;

        /** \brief Height of input depth image. */
        int rows_;
        /** \brief Width of input depth image. */
        int cols_;
        /** \brief Frame counter */
        int global_time_;

        /** \brief Truncation threshold for depth image for ICP step */
        float max_icp_distance_;

        /** \brief Intrinsic parameters of depth camera. */
        float fx_, fy_, cx_, cy_;

        /** \brief Tsdf volume container. */
//        TsdfVolume::Ptr tsdf_volume_;
        ColorVolume::Ptr color_volume_;
                
        /** \brief Initial camera rotation in volume coo space. */
        Matrix3frm init_Rcam_;

        /** \brief Initial camera position in volume coo space. */
        Vector3f   init_tcam_;

        /** \brief array with IPC iteration numbers for each pyramid level */
        int icp_iterations_[LEVELS];
        /** \brief distance threshold in correspondences filtering */
        float  distThres_;
        /** \brief angle threshold in correspondences filtering. Represents max sine of angle between normals. */
        float angleThres_;
        
        /** \brief Depth pyramid. */
        std::vector<DepthMap> depths_curr_;
        /** \brief Vertex maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_curr_;
        /** \brief Normal maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_curr_;

        /** \brief Vertex maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_prev_;
        /** \brief Normal maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_prev_;
                
        /** \brief Vertex maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> vmaps_curr_;
        /** \brief Normal maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> nmaps_curr_;

        //sema
         MapArr vmap_save, nmap_save;
         bool isICPfail;

        /** \brief Array of buffers with ICP correspondences for each pyramid level. */
        std::vector<CorespMap> coresps_;
        
        /** \brief Buffer for storing scaled depth image */
        DeviceArray2D<float> depthRawScaled_;
        
        /** \brief Temporary buffer for ICP */
        DeviceArray2D<double> gbuf_;
        /** \brief Buffer to store MLS matrix. */
        DeviceArray<double> sumbuf_;

        /** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Matrix3frm> rmats_;
        
        /** \brief Array of camera translations for each moment of time. */
        std::vector<Vector3f>   tvecs_;

        /** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Matrix3frm> rmats_icp, rmats_icp_nl;

        /** \brief Array of camera translations for each moment of time. */
        std::vector<Vector3f>   tvecs_icp, tvecs_icp_nl;

        /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
        float integration_metric_threshold_;
        
        /** \brief Allocates all GPU internal buffers.
          * \param[in] rows_arg
          * \param[in] cols_arg          
          */
        void
        allocateBufffers (int rows_arg, int cols_arg);

//        /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
//          */
//        void
//        reset ();

        //SEMA
        void reset_plane_coeffs();

        View raycast_view;

        Matrix3frm stable_Rcam_;
        Vector3f   stable_tcam_;

        float scale_factor;
    public:
        //SEMA
        void setCameraPose(Matrix3frm R, Vector3f t);
        void init_rmats_icp();
        void setScaleFactor(float d){scale_factor = d;}

        void addCurrentPtCloud(View& view);

        void revert_rmats_();

    };
  }
};

#endif /* PCL_KINFU_KINFUTRACKER_HPP_ */
