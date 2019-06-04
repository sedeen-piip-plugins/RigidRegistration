/*=============================================================================
 *
 *  Copyright (c) 2019 Sunnybrook Research Institute
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 *  Author: Rushin Shojaii
 *
 *  RigidRegistration is a Plugin (DLL) for Sedeen Viewer
 *  It uses DenseSIFT features for source and target images
 *  and register them by minimzing the distance between the features.
 *  inputs and outputs are through Sedeen Viewer.
 *  
 *=============================================================================*/

// Primary header
#include "./RigidRegistration.h"
#include "./tinyxml2.h"

#include <omp.h>  
extern int parallelism_enabled;

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <math.h>

// Poco header needed for the macros below 
#include <Poco/ClassLibrary.h>

#define PI 3.14159265

// Declare that this object has AlgorithmBase subclasses
//  and declare each of those sub-classes
POCO_BEGIN_MANIFEST(sedeen::algorithm::AlgorithmBase)
POCO_EXPORT_CLASS(sedeen::algorithm::RigidRegistration)
POCO_END_MANIFEST


using namespace cv;
using namespace std;
//using namespace boost;
using namespace tinyxml2;



namespace sedeen {
namespace algorithm {


	RigidRegistration::RigidRegistration()
		: output_text_(),
		useInitTrans_(),
		featurePrcnt_(),
		image_list_(),
		solver_(),
		ptr_F_(),
		output_image_(),
		t_dense_sift_(),
		tparam_(),
		elapsedTime_(),
		display_area_(),
		metric_(),
		sigma_(),
		maxIteration_(),
		epsilon_(),
		transOptimStep_(),
		rotOptimStep_(),
		scaleOptimStep_(),
		result_(),
		saveImage_(),
		sedeen_transform_(),
		reset_imageinfo_(),
		first_init_(true),
		channel_factory_(nullptr),
		output_factory_(nullptr) 		{	}


		/*************************************************/
		/***** init ******************************************/
		void RigidRegistration::init(const image::ImageHandle& image)
		{
			if (isNull(image)) return;

			//
			// bind algorithm members to UI and initialize their properties
			//
			// Current image list
			
			if (solver_ != nullptr)
				solver_.release();
			
			image_list_ = createImageListParameter(*this);		//Rushin
		
			
			solver_ = cv::DownhillSolver::create();
			//			solver_ = cv::ConjGradSolver::create(); // tried this and it didn't work well
			ptr_F_ = cv::makePtr<DenseSIFTRigidObjFunc>();      // Rushin: a pointer to DenseSIFTRigidObjFunc class
			float initparam[4] = { 0.0, 0.0, 0.0, 1.0 };
			tparam_ = cv::Mat(1, 4, CV_32F, initparam);         // Rushin: Holds the trandformation paramaters of source image
			//tparam_ = cv::Mat(1, 4, CV_64FC1, initparam);
	//		image_list_.setInfo(1, reset_imageinfo_);
			
	
			// Bind system parameter for current view
			display_area_ = createDisplayAreaParameter(*this);

			std::vector<std::string> options(4);
			options[0] = "Select";
			options[1] = "DenseSIFT";
			options[2] = "MI";
			options[3] = "ICP";
			metric_ = createOptionParameter(*this,
				"Registration type",
				"Regid registration type",
				1,        // default selection
				options,
				false); // list of all options

			std::vector<std::string> yesNoOptions(2);
			yesNoOptions[0] = "Yes";
			yesNoOptions[1] = "No";
			useInitTrans_ = createOptionParameter(*this,
				"Use initial Values",
				"Use manual transformation as initial values",
				0,
				yesNoOptions,
				false);

			switch (metric_)
			{
			case (1) :  // Dense SIFT
			{
				/*				useInitTrans_ = createBoolParameter(*this,
				"Use init Values",
				"Use manual transformation as initial values",
				bool(1),
				true);
				*/
				sigma_ = createDoubleParameter(*this,
					"Sigma",
					"Sigma value for gaussian filter",
					double(4.0),
					double(0.1),
					double(100.0),
					false);
				featurePrcnt_ = createIntegerParameter(*this,
					"Density of features (%)",
					"Density of SIFT features (%), select 100 for dense SIFT",
					int(20),  // default value,
					1,          // minimum value
					100,		// maximum value
					false);
				maxIteration_ = createIntegerParameter(*this,
					"Max. Iterations",
					"Maximum optimizer iterations",
					int(10000),  // default value,
					1,          // minimum value
					1000000,	// maximum value
					false);
				epsilon_ = createDoubleParameter(*this,
					"Epsilon",
					"Desired accuray or change in parameters",
					double(8),
					double(0),
					double(100),
					false);
				transOptimStep_ = createIntegerParameter(*this,
					"Translation step",
					"Initial translation step for optimizer",
					int(10),  // default value,
					0,          // minimum value
					10000,		// maximum value
					false);
				rotOptimStep_ = createIntegerParameter(*this,
					"Rotation step",
					"Initial rotation step for optimizer",
					int(5),  // default value,
					0,          // minimum value
					360,		// maximum value
					false);
				scaleOptimStep_ = createDoubleParameter(*this,
					"Scale step",
					"Initial scale step for optimizer",
					double(0.01),
					double(0.00),
					double(5.00),
					false);

			}
			case (2) :	//Rushin: You can add MI parameters here in future
			{

			}
			case (3) :	////Rushin: You can add ICP parameters here in future
			{

			}

			}

			saveImage_ = createOptionParameter(*this,
				"Save image",
				"Saves the transformed image if the size is less than 512x512",
				1,        // default selection
				yesNoOptions,
				false); // list of all options

			// Bind text and image results
			output_text_ = createTextResult(*this, "Text Result");
			result_ = createImageResult(*this, "Registration Result");
		}


		void RigidRegistration::resetParameters()
		{
			delete ptr_F_;
			if (t_image_ != nullptr) t_image_.reset();
			if (s_image_ != nullptr) s_image_.reset();
			if (ptr_F_ != nullptr) ptr_F_.release();
			if (solver_ != nullptr) solver_.release();
			
		}


		/*************************************************/
		/***** run *******************************************/
		void RigidRegistration::run()
		{
			// Has the list of images changed
			auto image_list_changed = image_list_.isChanged();
			// Has display area changed
			auto display_changed = display_area_.isChanged();

			DisplayRegion display_region = display_area_;
			// Get the region currently on screen
			image::tile::Compositor compositor(image()->getFactory());
			auto input = compositor.getImage(display_region.source_region, display_region.output_size);
			// Allocate output image of the same size & type as the input
			output_image_ = image::RawImage(input.size(), input);

	/*		auto dispScaledSize = display_region.output_size;
			auto dispSourceRect = display_region.source_region;
			image()->getFactory()->getLevelIndex(0);
	*/
		   // Rushin:  Build the Registration pipeline
			auto pipeline_changed = buildPipeline();

			// Update results 
			//////////////////////////////////////////
			// Rushin: I don't need to update results here. I just fill 
			// the results (transformation parameters) in sedeen viewer 
			// and sedden shows the transformed image in the viewer.  
			//////////////////////////////////////////
			if (pipeline_changed || display_changed || image_list_changed)
			{
				//result_.update(output_factory_, display_area_, *this);
				//result_.update(output_image_, output_image_.rect() );   //display_region.source_region);
			}
			////////////////////////////////////////////
			// Update the output text report
			// Rushin: reports the number of iterations of the optimizer and the elapsed time
			/////////////////////////////////////////////
			if (false == askedToStop())
			{
				auto report = generateReport();
				output_text_.sendText(report);
			}

		}


		///////////////////////////////////////////////////////////////////////////////////////
		// Rushin: in this version, I used the multi-level raw images from Sedeen DataServer.
		// GetRawImage returns the highest level (smallest) image for the image handle pointer im
		///////////////////////////////////////////////////////////////////////////////////////
		image::RawImage RigidRegistration::GetRawImage(const image::ImageHandle& im)
		{
			auto dataServer = createDataServer(im);
			auto nlevels = dataServer->getNumLevels();
			auto lowResRect = rect(im, nlevels - 1);
			auto rawImage = dataServer->getImage(nlevels - 1, lowResRect);
			if (rawImage.isNull())
				throw("Could not open the target image");

			return rawImage;
		}


		///////////////////////////////////////////////////////////////////////////////////////////////
		// Rushin: Tried to use Sedeen Display server to get the region of image on the display area, 
		// but this doesn't work for the target image, which is not active with the plugin.
		// -------- Created a ticket for path core to add a fundtion that returns the dipslay regions 
		// of all images loaded in Sedeen, but hasn't happened yet. ---------
		///////////////////////////////////////////////////////////////////////////////////////////////
		/*bool RigidRegistration::GetDisplayImage(const image::ImageHandle& im, image::RawImage rawImage)
		{
		auto dispServer = createDisplayServer(im);
		auto nlevels = dataServer->getNumLevels();
		auto lowResRect = rect(im, nlevels - 1);
		rawImage = dataServer->getImage(nlevels - 1, lowResRect);

		return true;
		}
		*/

		///////////////////////////////////////////////////////////////////////////////////////////////
		// Rushin: Tried to calculate and find the Display regions of the images, but it doesn't work. 
		// It adds more confusion with different transformation. The display regions of the images in 
		// the listexist in sedeen they just need to give us access to it.
		// -------- Created a ticket for path core to add a fundtion that returns the dipslay regions 
		// of all images loaded in Sedeen, but hasn't happened yet. ---------
		///////////////////////////////////////////////////////////////////////////////////////////////
		image::RawImage RigidRegistration::GetDisplayImage(const image::ImageHandle& im)
		{
			
			DisplayRegion display_region = display_area_;
			// Get the region currently on screen

			auto displayRegionScale = getDisplayScale(display_region);
			auto imDisplayIterator = getIterator(im, display_region);
			Point displayRegionPosition(imDisplayIterator.x(), imDisplayIterator.y());
			auto imageDisplaySize = display_region.output_size;
			Rect imageDisplayRect(displayRegionPosition, imageDisplaySize);
/*			auto input = image::getDisplayImage(im, imageDisplayRect, imageDisplaySize);
*/


			image::tile::Compositor compositor(im->getFactory());
	//		auto input = compositor.getImage(display_region.source_region, display_region.output_size);
			auto input = compositor.getImage(imageDisplayRect, display_region.output_size);
////			compositor.getImage(level, level_region);

//			auto dispServer = createDisplayServer(im);
//			auto input = dispServer->getImage(display_region.source_region, display_region.output_size);
		//	auto input = image::getDisplayImage(fromScreen-> im->getFactory()->, display_region.output_size, display_region)
			
			// Allocate output image of the same size & type as the input
/*			output_image_ = image::RawImage(input.size(), input);

			auto dispScaledSize = display_region.output_size;
			auto dispSourceRect = display_region.source_region;

			
			//			im->getFactory()->get
			image::getDisplayImage(im, )
			auto dispServer = createDisplayServer(im);
			auto dispSize = dispServer->getDimensions();
			auto dispRect = rect(im, nlevels - 1);
			rawImage = dataServer->getImage(nlevels - 1, lowResRect);
*/
			return input;
		}

		/////////////////////////////////////////////////////////////////////
		// Rushin: updates the region registration parametrs in an XML file
		/////////////////////////////////////////////////////////////////////
		bool RigidRegistration::UpdateXMLFile()
		{
			XMLDocument xml_doc;

			XMLDeclaration * decl = xml_doc.NewDeclaration();
			decl->SetValue("xml versin = 1.0");
			xml_doc.InsertFirstChild(decl);

			XMLNode * n_reg = xml_doc.NewElement("Registration"); //pRoot = xml_doc.NewElement("Root");
			xml_doc.LinkEndChild(n_reg); //xml_doc.InsertFirstChild(pRoot);

			XMLElement * pElement = xml_doc.NewElement("image");
			pElement->SetAttribute("identifier", regParamProcess_.xml_registration_file_.filename().string().data());

			XMLElement * innerElement = xml_doc.NewElement("target_image");
			innerElement->SetAttribute("identifier", regParamProcess_.target_id.data());

			XMLElement * regElement = xml_doc.NewElement("region");
			regElement->SetAttribute("region_id", "region identifier");
			XMLElement *BElement = xml_doc.NewElement("boundingbox");
			BElement->SetAttribute("x", regParamProcess_.curr_reg_param.region_boundingBox.x());
			BElement->SetAttribute("y", regParamProcess_.curr_reg_param.region_boundingBox.y());
			BElement->SetAttribute("center_x", regParamProcess_.curr_reg_param.region_boundingBox.center().getX());
			BElement->SetAttribute("center_y", regParamProcess_.curr_reg_param.region_boundingBox.center().getY());
			regElement->LinkEndChild(BElement);
			XMLElement *TElement = xml_doc.NewElement("transform");
			TElement->SetAttribute("translation_x", regParamProcess_.curr_reg_param.region_transform.translation().getX());
			TElement->SetAttribute("translation_y", regParamProcess_.curr_reg_param.region_transform.translation().getY());
			TElement->SetAttribute("rot_center_x", regParamProcess_.curr_reg_param.region_transform.center().getX());
			TElement->SetAttribute("rot_center_y", regParamProcess_.curr_reg_param.region_transform.center().getY());
			TElement->SetAttribute("rotation", regParamProcess_.curr_reg_param.region_transform.rotation());
			TElement->SetAttribute("scale_x", regParamProcess_.curr_reg_param.region_transform.scale().width());
			TElement->SetAttribute("scale_y", regParamProcess_.curr_reg_param.region_transform.scale().height());
			regElement->LinkEndChild(TElement);
			innerElement->InsertEndChild(regElement);

			pElement->LinkEndChild(innerElement);
			n_reg->LinkEndChild(pElement);

			XMLError eResult = xml_doc.SaveFile(regParamProcess_.xml_registration_file_.string().data(), false);
			if (eResult != XML_SUCCESS)
				printf("Error: %i\n", eResult);
			return true;
		}

		///////////////////////////////////////////////////////
		// Converts the transformatons from OpenCV to Sedeen
		///////////////////////////////////////////////////////
		bool RigidRegistration::OpenCVtoSedeenTransform()
		{
			double teta = (360 - tparam_.at<float>(0, 2)) * PI / 180; //radian
			double dy = -tparam_.at<float>(0, 0) - (-tparam_.at<float>(0, 0)*cos(teta) - tparam_.at<float>(0, 1)*sin(teta));
			double dx = -(tparam_.at<float>(0, 1) - (-tparam_.at<float>(0, 0)*sin(teta) + tparam_.at<float>(0, 1)*cos(teta)));

			auto full_x_diff = (image()->getMetaData()->get(image::IntegerTags::IMAGE_X_DIMENSION, 0)*s_image_props_.spacing.width() - t_image_->getMetaData()->get(image::IntegerTags::IMAGE_X_DIMENSION, 0)*t_image_props_.spacing.width()) / 2.;
			auto full_y_diff = (image()->getMetaData()->get(image::IntegerTags::IMAGE_Y_DIMENSION, 0)*s_image_props_.spacing.height() - t_image_->getMetaData()->get(image::IntegerTags::IMAGE_Y_DIMENSION, 0)*t_image_props_.spacing.height()) / 2.;

			sedeen_transform_.setRotation(360 - tparam_.at<float>(0, 2));
			sedeen_transform_.setTranslation(-(dx * s_image_props_.spacing.width()*(s_image_props_.nlevels - 1) * 4) - full_x_diff, -(dy * s_image_props_.spacing.height()*(s_image_props_.nlevels - 1) * 4) - full_y_diff);
			sedeen_transform_.setScale(tparam_.at<float>(0, 3), tparam_.at<float>(0, 3));

			//	double cX = image()->getMetaData()->get(image::IntegerTags::IMAGE_X_DIMENSION, 0)*s_image_props_.spacing.width()/2;
			//	double cY = image()->getMetaData()->get(image::IntegerTags::IMAGE_Y_DIMENSION, 0)*s_image_props_.spacing.height()/2;
			double cX = s_image_props_.centre.getX();
			double cY = s_image_props_.centre.getY();
			cX = cX - (dx * 4 * (s_image_props_.nlevels - 1) * (s_image_props_.spacing.width()));
			cY = cY - (dy * 4 * (s_image_props_.nlevels - 1) * (s_image_props_.spacing.height()));
			sedeen_transform_.setCenter(cX, cY);

			/*	SRTTransform c_transform(0, 0, tparam_.at<float>(0, 3), tparam_.at<float>(0, 3), 0., 0., 0.);
			c_transform.setTranslation(dx*s_image_props_.spacing.width(), dy*s_image_props_.spacing.height());
			*/
			
/*			cv::FileStorage fname("D:/Documents/Research/Sedeen/Output/RigidRegParamInOpenCVSedeen.ext", cv::FileStorage::WRITE);
			fname << "Rigid Rotation" << tparam_.at<float>(0, 2) << "X" << tparam_.at<float>(0, 0) << "Y" << tparam_.at<float>(0, 1)
				<< "scaleX" << tparam_.at<float>(0, 3);
			fname << "elapsed Time " << elapsedTime_; // << "number of iterations" << ptr_F_->count;
			fname.release();
*/
			return true;
		}

		///////////////////////////////////////////////////////
		// Converts the transformatons from Sedeen to OpenCV
		///////////////////////////////////////////////////////
		bool RigidRegistration::SedeentoOpenCVtoTransform()
		{
			tparam_.at<float>(0, 0) = s_image_props_.sedeen_transform.translation().getX() / (s_image_props_.spacing.width() * 4 * (s_image_props_.nlevels - 1));  //pixels for OpenCV
			tparam_.at<float>(0, 1) = s_image_props_.sedeen_transform.translation().getY() / (s_image_props_.spacing.height() * 4 * (s_image_props_.nlevels - 1));
			tparam_.at<float>(0, 2) = 360 - s_image_props_.sedeen_transform.rotation();
			tparam_.at<float>(0, 3) = s_image_props_.sedeen_transform.scale().width();

/*			cv::FileStorage fname("D:/Documents/Research/Sedeen/Output/RigidRegParamInSedeenOpenCV.ext", cv::FileStorage::WRITE);
			fname << "Rigid Rotation" << tparam_.at<float>(0, 2) << "X" << tparam_.at<float>(0, 0) << "Y" << tparam_.at<float>(0, 1)
				<< "scaleX" << tparam_.at<float>(0, 3);
			fname << "spacing " << s_image_props_.spacing.width(); // << "number of iterations" << ptr_F_->count;
			fname.release();
*/
			return true;
		}

		/////////////////////////////////////////////////////////
		// Returns all image info required for the plugin 
		/////////////////////////////////////////////////////////
		ImageProperties RigidRegistration::GetImageInfo(const image::ImageHandle& im, sedeen::algorithm::ImageInfo *imageinfo)
		{
			ImageProperties imProps;

			//auto imageinfo = image_list_.info(index);     //image_list_.indexOf(image()));
			imProps.sedeen_transform = imageinfo->transform;
			imProps.opacity = imageinfo->opacity;
			imProps.visibility = imageinfo->visible;
			imProps.location = imageinfo->location;

			//auto dataServer = createDataServer(im);
			//imProps.nlevels = dataServer->getNumLevels();


			int im_width = im->getMetaData()->get(image::IntegerTags::IMAGE_X_DIMENSION, 0);  // pixel
			int im_height = im->getMetaData()->get(image::IntegerTags::IMAGE_Y_DIMENSION, 0);
			imProps.size = sedeen::Size(im_width, im_height);

			double x_spacing = 1;	//mm with no pixel spacing
			double y_spacing = 1;
			if (im->getMetaData()->has(image::DoubleTags::PIXEL_SIZE_X))
				x_spacing = im->getMetaData()->get(image::DoubleTags::PIXEL_SIZE_X, 0) / 1000; //mm
			if (im->getMetaData()->has(image::DoubleTags::PIXEL_SIZE_Y))
				y_spacing = im->getMetaData()->get(image::DoubleTags::PIXEL_SIZE_Y, 0) / 1000;
			imProps.spacing = sedeen::SizeF(x_spacing, y_spacing);

			double x_centre = im->getMetaData()->get(image::DoubleTags::IMAGE_CENTRE_X, 0) / 2;   //IMAGE_CENTRE_X is in mm
			double y_centre = im->getMetaData()->get(image::DoubleTags::IMAGE_CENTRE_Y, 0) / 2;	  //IMAGE_CENTRE_Y is in mm
			imProps.centre = sedeen::PointF(x_centre, y_centre);

/*			cv::FileStorage fname("D:/Documents/Research/Sedeen/Output/RigidRegParamInGetImageInfo.ext", cv::FileStorage::WRITE);
			fname << "Rigid Rotation" << imProps.sedeen_transform.rotation() << "X" << imProps.sedeen_transform.translation().getX() << "Y" << imProps.sedeen_transform.translation().getY()
				<< "scaleX" << imProps.sedeen_transform.scale().width();
			fname << "spacing " << imProps.spacing.width(); // << "number of iterations" << ptr_F_->count;
			fname.release();
*/
			return imProps;
		}

		bool RigidRegistration::SetImageInfo(int index)
		{
			auto imageinfo = image_list_.info(index);
			imageinfo.transform.setCenter(s_image_props_.centre);
			imageinfo.transform.setRotation(sedeen_transform_.rotation());
			imageinfo.transform.setTranslation(sedeen_transform_.translation());
			imageinfo.transform.setScale(sedeen_transform_.scale());
			imageinfo.transform.setCenter(sedeen_transform_.center());
			image_list_.setInfo(index, imageinfo);

			return true;
		}

		std::vector<cv::KeyPoint> RigidRegistration::FindKeyPoints(int rows, int cols, int spacingfactor)
		{
			std::vector<cv::KeyPoint> kps;
			int step = uint((100/featurePrcnt_)*spacingfactor); // 1 for dense sift
			int nfeat = 0;

//#pragma omp parallel for 
			for (int k = step; k < rows*spacingfactor - (2*step); k += step)
			{
				for (int j = step; j < cols*spacingfactor - (2*step); j += step)
				{
					kps.push_back(cv::KeyPoint(float(j), float(k), 2.0));  //float(step)));
					nfeat = nfeat + 1;
				}
			}
			ptr_F_->nfeature_ = nfeat;

			return kps;
		}

		/*************************************************/
		/***** buildPipeline ***********************************/
		bool RigidRegistration::buildPipeline()
		{
			// For Cache, FilterFactory, ChannelSelect, Threshold, etc...
			using namespace image::tile;
			bool pipeline_changed = false;
			
			if (metric_ == 1 && image_list_.isChanged() || useInitTrans_.isChanged())
			{
				if (image_list_.count() < 2)
					throw std::runtime_error("Source and target images must be loaded.");
				if (image_list_.indexOf(image()) == 0)
					throw std::runtime_error("Source image must be different from target (first) image.");

				auto t_imageinfo = image_list_.info(0);
				auto t_location = t_imageinfo.location;
				//				auto t_opacity = t_imageinfo.opacity;
				//				auto t_visibility = t_imageinfo.visible;
				auto t_transform = t_imageinfo.transform;
				auto t_imgopener = image::createImageOpener();
				t_image_ = t_imgopener->open(file::Location(t_location));
				if (!t_image_)
					throw std::runtime_error("Could not open the target image: " + file::Location(t_location).getFilename());
				t_image_props_ = GetImageInfo(t_image_, &t_imageinfo);

				auto s_index = image_list_.indexOf(image());
				auto s_imageinfo = image_list_.info(s_index);
				auto s_location = s_imageinfo.location;
				//				auto s_opacity = s_imageinfo.opacity;
				//				auto s_visibility = s_imageinfo.visible;
				auto s_transform = s_imageinfo.transform;
				auto s_imgopener = image::createImageOpener();
				s_image_ = s_imgopener->open(file::Location(s_location));
				if (!s_image_)
					throw std::runtime_error("Could not open the source image: " + file::Location(s_location).getFilename());
				s_image_props_ = GetImageInfo(s_image_, &s_imageinfo);

				if (first_init_ == true)
				{
					first_init_ = false;
					reset_imageinfo_ = s_imageinfo;
				}


				/******** Loading Target Image **********/
/*				auto t_input = image::loadImage(t_location);
				auto t_output = image::RawImage(t_input.size(), t_input);
				auto t_input_mat = image::toOpenCV(t_input, true);
				auto t_output_mat = image::toOpenCV(t_output, true);
				if (t_input_mat.channels() > 1)
					cv::cvtColor(t_input_mat, t_output_mat, cv::COLOR_BGR2GRAY);
				//				cv::extractChannel(t_input_mat, t_output_mat, 2);
*/
				auto t_raw = GetRawImage(t_image_);
		//		auto t_raw = GetDisplayImage(t_image_);
		t_raw.save("D:/Documents/Research/Sedeen/TestImages/MRI_Histo_pigHeart/SPIE_demo/TdispImage.tif");
				auto t_input_mat = image::toOpenCV(t_raw, true);
				cv::Mat t_output_mat(t_input_mat.size(), CV_8U);
				if (t_input_mat.channels() > 2)
					cv::cvtColor(t_input_mat, t_output_mat, cv::COLOR_BGR2GRAY);


				/*** Getting the region on target image for registration ***/
				Session t_session = Session(t_location.c_str());		// Target image session
				std::vector<GraphicDescription> t_graph_vector;
				if (t_session.loadFromFile() == TRUE)
					t_graph_vector = t_session.getGraphics();


				/******** Loading Source Image **********/
/*				auto s_input = image::loadImage(s_location);
				auto s_output = image::RawImage(s_input.size(), s_input);
				auto s_input_mat = image::toOpenCV(s_input, true);
				auto s_output_mat = image::toOpenCV(s_output, true);
				if (s_input_mat.channels() > 1)
					cv::cvtColor(s_input_mat, s_output_mat, cv::COLOR_BGR2GRAY);
				//				cv::extractChannel(s_input_mat, s_output_mat, 2);
*/
				auto s_raw = GetRawImage(s_image_);
		//		auto s_raw = GetDisplayImage(s_image_);
		s_raw.save("D:/Documents/Research/Sedeen/TestImages/MRI_Histo_pigHeart/SPIE_demo/SdispImage.tif");
				auto s_input_mat = image::toOpenCV(s_raw, true);
				cv::Mat s_output_mat(s_input_mat.size(), CV_8U);
				if (s_input_mat.channels() > 1)
					cv::cvtColor(s_input_mat, s_output_mat, cv::COLOR_BGR2GRAY);

//				t_image_props_ = GetImageInfo(t_image_, &t_imageinfo);
//				s_image_props_ = GetImageInfo(s_image_, &s_imageinfo);
				auto xdiff = (t_image_props_.size.width() - s_image_props_.size.width()) / 2;
				auto ydiff = (t_image_props_.size.height() - s_image_props_.size.height()) / 2;


				int nleyers = 1;
//#pragma omp parallel for
				for (int i = 0; i <= nleyers - 1; i++)
				{

					/***********************************************************/
					/*****  Optimization using Nelder Mead simplex method ******/
					/***********************************************************/
					ptr_F_->s_image_ = s_output_mat;
					ptr_F_->sigma = sigma_;

					/************************************************************/
					/**** Creating key points excluding the boundary pixels *****/
					/************************************************************/
					ptr_F_->t_kps_ = FindKeyPoints(min(t_output_mat.rows, s_output_mat.rows), min(t_output_mat.cols, s_output_mat.cols), (s_image_props_.spacing.width()/t_image_props_.spacing.width()));
					ptr_F_->s_kps_ = FindKeyPoints(min(t_output_mat.rows, s_output_mat.rows), min(t_output_mat.cols, s_output_mat.cols), 1);



					/***************************************************************/
					/*	// Rushin: For images with different size define the new key points here
					for (int i = step; i < s_input_mat.rows - step; i += step)
					{
					for (int j = step; j < s_input_mat.cols - step; j += step)
					kps.push_back(cv::KeyPoint(float(j), float(i), float(step)));
					}
					
					*/	/***************************************************************/
					/*	  // Writing the key points to check the values
					///////////// Rushin: Not needed any more, this was just to test the keypoints
					std::vector<cv::Point2f> points;
					std::vector<cv::KeyPoint>::iterator it;
					for (it = kps.begin(); it != kps.end(); it++)
					points.push_back(it->pt);
					cv::Mat pointmatrix(points);
					cv::FileStorage fname("C:/Program Files (x86)/Sedeen Viewer SDK/v/msvc2012/examples/plugins/Registration/Output/keys.ext", cv::FileStorage::WRITE);
					fname << "kps" << pointmatrix;
					fname.release();
					*/

					/************************************************************/
					/****** Computing target SIFT features for the dense key points
					/************************************************************/
					cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> t_sift = cv::xfeatures2d::SiftDescriptorExtractor::create(ptr_F_->nfeature_, 3, 0.00, 255.0, sigma_); // 128, 3, 100., 10., 1.6);
					t_sift->compute(t_output_mat, ptr_F_->t_kps_, t_dense_sift_);
					ptr_F_->t_dense_sift_ = t_dense_sift_;
					 
					/*	  // Writing the descriptors to check the values   // Rushin needed any more
					cv::FileStorage fname("C:/Program Files (x86)/Sedeen Viewer SDK/v/msvc2012/examples/plugins/Registration/Output/t_sift.ext", cv::FileStorage::WRITE);
					fname << "t_dense_sift" << t_dense_sift;
					fname.release();
					*/


					////tparam are filled with the user input parameters
					cv::Mat optimstep = (cv::Mat_<double>(1, 4) << transOptimStep_, transOptimStep_, rotOptimStep_, scaleOptimStep_);    //10, 10, 1, 1.1);
					float initparam[4] = { 0.0, 0.0, 0.0, 1.0 };
					tparam_.at<float>(0, 0) = 0.0;
					tparam_.at<float>(0, 1) = 0.0;
					tparam_.at<float>(0, 2) = 0;
					tparam_.at<float>(0, 3) = 1.0;

					if (useInitTrans_ == 0) //yes
					{
						tparam_.at<float>(0, 0) = s_transform.translation().getX();// 
						tparam_.at<float>(0, 1) = s_transform.translation().getY();// 
						tparam_.at<float>(0, 2) = 360 - s_transform.rotation();
						tparam_.at<float>(0, 3) = s_transform.scale().width();
						
					}
					
					double etalon_res = 0.0;
					solver_->setFunction(ptr_F_);
					solver_->setInitStep(optimstep);
					solver_->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIteration_, epsilon_)); //100000, 10000));

					//////// Rushin: wrote this just for checking the parameters
					//cv::FileStorage fname("D:/Documents/Research/Sedeen/Output/RigidRegParam.ext", cv::FileStorage::WRITE);
					//				fname << "InitialRotation" << s_transform.rotation() << "tparamRot" << tparam_.at<float>(0, 2) << "X" << tparam_.at<float>(0, 0) << "Y" << tparam_.at<float>(0, 1)
					//					<< "scaleX" << tparam_.at<float>(0, 3) << "RotCenter X" << rotCenter.getX() << "RotCenter Y" << rotCenter.getY();

					auto e1 = cv::getTickCount();
					double res = solver_->minimize(tparam_);
					auto e2 = cv::getTickCount();
					elapsedTime_ = (e2 - e1) / cv::getTickFrequency();

					//tparam_.at<float>(0, 4) = tparam_.at<float>(0, 3);  // Rushin: opencv only allows for isotropic scale factor

					// Rushin: commented out because it is just for test
/*					fname << "Res" << res << "Rigid Rotation" << tparam_.at<float>(0, 2) << "X" << tparam_.at<float>(0, 0) << "Y" << tparam_.at<float>(0, 1)
						<< "scaleX" << tparam_.at<float>(0, 3);
					fname << "elapsed Time " << elapsedTime_; // << "number of iterations" << ptr_F_->count;
					fname.release();
*/
					float spacing = s_image_props_.spacing.width(); // 1;
				
					double teta = (360 - tparam_.at<float>(0, 2)) * PI / 180; //radian
					float dx = -tparam_.at<float>(0, 0) - (-tparam_.at<float>(0, 0)*cos(teta) - tparam_.at<float>(0, 1)*sin(teta));
					float dy = -(tparam_.at<float>(0, 1) - (-tparam_.at<float>(0, 0)*sin(teta) + tparam_.at<float>(0, 1)*cos(teta)));
					auto tx = dx - xdiff;
					auto ty = dy - ydiff;
					s_imageinfo.transform.setTranslation(tx*spacing, ty*spacing);
					PointF cntr(((s_input_mat.cols / 2.) + dx)*4*spacing*1000, ((s_input_mat.rows / 2.) + dy)*spacing*1000);
					s_imageinfo.transform.setRotation((360 - tparam_.at<float>(0, 2))); //, cntr);
					s_imageinfo.transform.setScale(tparam_.at<float>(0, 3), tparam_.at<float>(0, 3));

					image_list_.setInfo(s_index, s_imageinfo);
				}

				// Rushin: just to test the values
				/*				fname << "RotCenX" << s_transform.center().getX() << "RotCenY" << s_transform.center().getY();
				fname << "deltax" << dx << "deltay" << dy << "centerx" << cx << "centery" << cy;
				*/
				if (saveImage_ == 0) //Save transformed image
				{
					cv::Mat dst;
					//					std::string filename;
					cv::Point2f pc((s_input_mat.cols / 2.) + tparam_.at<float>(0, 0), (s_input_mat.rows / 2.) + tparam_.at<float>(0, 1));
					cv::Mat r = cv::getRotationMatrix2D(pc, tparam_.at<float>(0, 2), tparam_.at<float>(0, 3));
					cv::warpAffine(s_input_mat, dst, r, s_input_mat.size(), 1, cv::BORDER_CONSTANT, cv::Scalar(215.0, 215.0, 215.0));

					auto regimagefile = s_imageinfo.location;
					regimagefile.insert(regimagefile.begin() + regimagefile.find('.'), '_');
					regimagefile.insert(regimagefile.begin() + regimagefile.find('.'), 'r');
					regimagefile.insert(regimagefile.begin() + regimagefile.find('.'), 'e');
					regimagefile.insert(regimagefile.begin() + regimagefile.find('.'), 'g');

					output_image_ = image::fromOpenCV(dst, false);
					output_image_.save(regimagefile);
					/////// Rushin: uncommenting this line allows to save the transformed image the name of the file needs to be changed to a proper one 
					//					output_image_.save("D:/Documents/Research/Sedeen/TestImages/AWS/Registered/Reg_Image.jpg");
				}
				// testing
				/*				fname << "pcX" << (s_input_mat.cols / 2.) + tparam_.at<float>(0, 0) << "pcY" << (s_input_mat.rows / 2.) + tparam_.at<float>(0, 1);
				fname.release();
				*/
				
//				UpdateXMLFile();  // Rushin: Uncomment when the display regions are are used for registration 
				pipeline_changed = true;

			}//end if for two loaded images

			return pipeline_changed;
		}


		/****************************************************************************/
		/***** generateReport *******************************************************/
		// Rushin: Reports the number of iterations of the optimizer and the elapsed time
		///////////////////////////////////////////////////////////////////////////
		std::string RigidRegistration::generateReport() const{

			std::ostringstream ss;
			ss << std::left << std::setfill(' ') << std::setw(15);
			ss << std::left << std::setfill(' ') << std::setw(20);
			ss << "Number of iterations: " << std::setprecision(2) << ptr_F_->count_ << std::endl;
			ss << "Elapsed time: " << std::setprecision(2) << this->elapsedTime_ << " sec" << std::endl;
			ss << std::endl;
			ptr_F_->count_ = 0;
			
			return ss.str();
		}


		/*****************************************************************************/
		/******************* DenseSIFTRigidObjFunc Memeber Functions *****************/
		/*****************************************************************************/
		DenseSIFTRigidObjFunc::DenseSIFTRigidObjFunc()
		{
			count_ = 0;
//			fname_.open("D:/Documents/Research/Sedeen/TestImages/MRI_Histo_pigHeart/calcnfeat.ext", cv::FileStorage::WRITE);
		}

		DenseSIFTRigidObjFunc::~DenseSIFTRigidObjFunc()
		{
//			fname_.release();
		}

		double DenseSIFTRigidObjFunc::calc(const double* tparam) const
		{
			count_ += 1;
			cv::Point2f center((s_image_.cols / 2.0) + tparam[0], (s_image_.rows / 2.0) + tparam[1]);  //(--x, -y)

			cv::Mat trans = cv::getRotationMatrix2D(center, tparam[2], tparam[3]);
			cv::Mat s_image_trans;
			cv::warpAffine(s_image_, s_image_trans, trans, s_image_.size(), 1, cv::BORDER_CONSTANT, cv::Scalar(215.0, 215.0, 215.0));

			cv::Mat s_dense_sift;
			std::vector<cv::KeyPoint> kps;
			kps = s_kps_;
			std::vector<cv::KeyPoint> tkps;
			tkps = t_kps_;

			/************************************************************/
			/****** Computing source SIFT features for the dense key points
			/************************************************************/
			cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> s_sift = cv::xfeatures2d::SiftDescriptorExtractor::create(nfeature_, 3, 0.00, 255.0, sigma); // 128, 3, 100., 10., 1.6);
			s_sift->compute(s_image_trans, kps, s_dense_sift);

			//Rushin: testing the values
/*			fname_ << "norm" << cv::norm(t_dense_sift_, s_dense_sift, cv::NORM_L2);
			for (std::vector<cv::KeyPoint>::iterator it = kps.begin(); it != kps.end(); ++it)
				fname_ << "kps" << it->pt;
			for (std::vector<cv::KeyPoint>::iterator it = tkps.begin(); it != tkps.end(); ++it)
				fname_ << "t_kps" << it->pt;
*/
			return cv::norm(t_dense_sift_, s_dense_sift, cv::NORM_L2);
		}

	} // namespace algorithm
} // namespace sedeen



/*************************************************/
/* Unit Test ***********************************/
/*int main(int argc, char**argv)
{
	namespace si = sedeen::image;
	auto opener = si::createImageOpener();

	namespace sf = sedeen::file;

	assert(argc != 1); //requires to image files to register

	auto timage = opener->open(sf::Location(argv[0]));
	auto simage = opener->open(sf::Location(argv[1]));
	namespace sa = sedeen::algorithm;
	sa::ConsoleIOFactory iofactory;

	sa::RigidRegistration registration;

	registration.attach(timage);
	registration.connect(iofactory);

	//	iofactory.setInteger(0, 20);
	iofactory.setInteger(1, 20);
	//
	registration.execute();

	//	auto a = iofactory.getInteger(0);
	//	assert(20 == a);
	//	auto b = iofactory.getInteger(1);
	//	assert(40 == b);
	//
	return 0;

}
*/