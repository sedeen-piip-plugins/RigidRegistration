PROJECT( RigidRegistration )
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

## SET(GCC_PARALLEL_COMPILE_FLAGS  "-Qpar")   ##"-fprofile-arcs -ftest-coverage")
## SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

##
## Load the Sedeen dependencies
SET( CMAKE_PREFIX_PATH "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012" )
FIND_PACKAGE( SEDEENSDK REQUIRED "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012" )

## Load Boost dependency
SET (CMAKE_PREFIX_PATH "C:/Boost" )
FIND_PACKAGE( BOOST REQUIRED "C:/Boost" )


## Load OpenCV dependency
SET (CMAKE_PREFIX_PATH "D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install" )
FIND_PACKAGE( OPENCV REQUIRED "D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install" )

INCLUDE_DIRECTORIES( "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/include" 
		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include" 
		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include/opencv" 
		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include/opencv2"
		"C:/Boost/include/boost-1_55")

LINK_DIRECTORIES( "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/lib" 
		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/x86/vc12/lib"
		"C:/Boost/lib/i386" 
		"C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/bin")

##
## Build the code into a module library
ADD_LIBRARY( RigidRegistration MODULE RigidRegistration.cpp tinyxml2.cpp RigidRegistration.h tinyxml2.h)

# Link the library against the Sedeen libraries
# NOTE: The QT libraries must be linked first.
TARGET_LINK_LIBRARIES( RigidRegistration image.lib global.lib algorithm.lib archive.lib 
		bindings_opencv.lib opencv_core310.lib opencv_imgproc310.lib opencv_ts310.lib 
		opencv_features2d310.lib opencv_xfeatures2d310.lib) 

## 
##
## Install the plugin in the sedeen plugins directory
##IF( ${PATHCORE_FOUND} )
  INSTALL( TARGETS RigidRegistration LIBRARY DESTINATION "C:/Program Files (x86)/Sedeen Viewer/plugins/cpp/piip/RigidRegistration" )
##ENDIF()
