/******************************************************/
// Author: Rushin Shojaii
// 
// RigidRegistration is a Plugin (DLL) for Sedeen Viewer 
// It uses DenseSIFT features for source and target images 
// and register them by minimzing the distance between the features.
// inputs and outputs are through Sedeen Viewer.
//
/******************************************************/


#ifndef SEDEEN_SRC_PLUGINS_RIGIDREGISTRATION_RIGIDREGISTRATION_H
#define SEDEEN_SRC_PLUGINS_RIGIDREGISTRATION_RIGIDREGISTRATION_H

#include <numeric>
#include <list>

// DPTK headers - a minimal set 
#include "algorithm/AlgorithmBase.h"
#include "algorithm/Parameters.h"
#include "algorithm/Results.h"
#include "algorithm/ImageListParameter.h"	
#include "image/io/DataServer.h"

// DPTK headers
#include "Algorithm.h"
#include "Archive.h"
#include "BindingsOpenCV.h"
#include "Geometry.h"
#include "Global.h"
#include "Image.h"
#include <global/geometry/SRTTransform.h>
#include <global/geometry/PointF.h>

// Include OpenCV headers
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/optim.hpp>

// include Boost headers
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range/irange.hpp>
#include <boost/filesystem.hpp>

namespace pt = boost::property_tree;

namespace sedeen {
	namespace algorithm {


		/***********************************************************************************/
		/**** This struct holds the image properties
		/***********************************************************************************/
		struct ImageProperties
		{
			std::basic_string <char> location;
			int nlevels;
			sedeen::SRTTransform sedeen_transform;
			int opacity;
			bool visibility;
			sedeen::SizeF	spacing;
			sedeen::Size size;
			sedeen::PointF centre;

			double x_centre;
			double y_centre;
		};



		/***********************************************************************************/
		/**** This struct holds the region's registration info for XML file
		/***********************************************************************************/
		struct RegionParameters
		{
			RectF region_boundingBox;
			PointF region_centre;
			double region_magnification;
			SRTTransform region_transform;
		};


		/***********************************************************************************/
		/**** This class manages the registration XML file
		/***********************************************************************************/
		class ManageRegParamXMLFile{
		public:
			std::string im_id;						//image/source filename
			sedeen::Size im_dimensions;				//image dimensions
			sedeen::SizeF im_pixelSize;			    //image resolution
			std::string target_id;                  // target filename
			RegionParameters curr_reg_param;
			RegionParameters nearest_reg_param;
			//std::list<RegionParameters> reg_param_list;
			pt::ptree xml_tree;

		public:
			boost::filesystem::path xml_registration_file_ = "";
			bool loadRegionParam();
			bool saveRegionParam(const std::string &filename);
			//		RegionParameters findNearestRegionParam();
			bool findNearestRegionParam(Rect *);
			bool addCurrentParamToList();
			bool updateXMLTree();
			bool deleteNodefromParamList(std::string nodeName);
			bool deleteNodefromTargetList(std::string nodeName);
		};



		/***********************************************************************************/
		/**** Er(teta, tx, ty, Sx, Sy) = Sigma || tSift(p) - sSift(p') ||             ******/
		/**** p = (x,y)                                                               ******/
		/**** p' = (xSxCos(teta) - ySin(teta) + tx , xSin(teta) + ySyCos(teta) + ty ) ******/
		/**** tparam includes: tx, ty, teta, Sx, Sy									  ******/
		/***********************************************************************************/
		class DenseSIFTRigidObjFunc : public cv::MinProblemSolver::Function
		{
		public:
			DenseSIFTRigidObjFunc();
			~DenseSIFTRigidObjFunc();
			cv::Mat t_dense_sift_;
			//		cv::Mat s_dense_sift_;
			mutable int count_;
			double sigma;
			int nfeature_;
			std::vector<cv::KeyPoint> s_kps_;
			std::vector<cv::KeyPoint> t_kps_;
			std::shared_ptr<cv::Mat> t_image_;
//			std::shared_ptr<cv::Mat> s_image_;
			cv::Mat s_image_;

			int getDims() const { return 4; }
			double calc(const double* tparam) const;
			mutable cv::FileStorage fname_;
			SizeF spacing_ratio_;
			SizeF size_diff_;
		};


		/***********************************************************************************/
		/****  Registration: public Algorithm
		/***********************************************************************************/
		//	class Registration : public algorithm::AlgorithmBase {
		class RigidRegistration : public AlgorithmBase {
		public:
			RigidRegistration();

		private:
			// virtual function
			virtual void run();
			virtual void init(const image::ImageHandle& image);
			//virtual void init(const image::ImageHandle& image, const image::ImageHandle& simage);

			/// Creates the pipeline with a cache
			/// \return TRUE if the pipeline has changed since the call to this function, FALSE otherwise
			image::RawImage GetRawImage(const image::ImageHandle& im);
			image::RawImage GetDisplayImage(const image::ImageHandle& im);

			bool UpdateXMLFile();

			std::shared_ptr<sedeen::image::Image> s_image_;
			std::shared_ptr<sedeen::image::Image> t_image_;
			ImageProperties s_image_props_;
			ImageProperties t_image_props_;
			int s_nlevels_;
			int t_nlevels_;
			bool SedeentoOpenCVtoTransform();
			bool OpenCVtoSedeenTransform();
			SRTTransform sedeen_transform_;
//			SRTTransform reset_trans_;
			ImageInfo reset_imageinfo_;
			mutable bool first_init_;
			std::vector<cv::KeyPoint> FindKeyPoints(int rows, int cols, int spacingfactor);
			//		RegionParameters *regParam_;
			ManageRegParamXMLFile regParamProcess_;
			ImageProperties GetImageInfo(const image::ImageHandle& im, sedeen::algorithm::ImageInfo *imageinfo);

			bool SetImageInfo(int index);
			void resetParameters();

			bool buildPipeline();
			std::string generateReport() const;

		private:
			TextResult output_text_;
			OptionParameter useInitTrans_;
			IntegerParameter featurePrcnt_;
			ImageListParameter image_list_;
			cv::Ptr<cv::DownhillSolver> solver_;
			//		cv::Ptr<cv::ConjGradSolver> solver_;
			cv::Ptr<DenseSIFTRigidObjFunc> ptr_F_;     // a pointer to DenseSIFTRigidObjFunc
			image::RawImage output_image_;
			cv::Mat t_dense_sift_;
			cv::Mat tparam_;
			double elapsedTime_;

			//algorithm::DisplayAreaParameter display_area_;
			DisplayAreaParameter display_area_;
			OptionParameter metric_;
			OptionParameter saveImage_;
			DoubleParameter sigma_;
			IntegerParameter maxIteration_;
			DoubleParameter epsilon_;
			IntegerParameter transOptimStep_;
			IntegerParameter rotOptimStep_;
			DoubleParameter scaleOptimStep_;
			ImageResult result_;


			/// The cached output of an intermediate stage in the pipeline
			std::shared_ptr<image::tile::Factory> channel_factory_;
			/// The cached output on the pipeline
			std::shared_ptr<image::tile::Factory> output_factory_;
		};


	} // namespace algorithm
} // namespace sedeen


#endif

