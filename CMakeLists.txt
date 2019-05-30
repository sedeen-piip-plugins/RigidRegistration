PROJECT( RigidRegistration )
CMAKE_MINIMUM_REQUIRED( VERSION 3.8 )
# Enable C++17 features
SET(CMAKE_CXX_STANDARD 17)

## SET(GCC_PARALLEL_COMPILE_FLAGS  "-Qpar")   ##"-fprofile-arcs -ftest-coverage")
## SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# Define project description variables
SET( DISPLAY_NAME_TEXT "RigidRegistration 5.4.1.20190327" CACHE STRING "Name of the plugin as it should be displayed in Sedeen Viewer")
SET( SUPPORT_URL_TEXT "http://pathcore.com/support/plugin/info/${PROJECT_NAME}" CACHE STRING "Location users can find help with the plugin" )
SET( DEVELOPER_TEXT "Sunnybrook Research Institute" CACHE STRING "Name of the author or organization that created the plugin" )

# Load the Sedeen dependencies
SET(PROGRAMFILESX86 "PROGRAMFILES\(X86\)")
FIND_PACKAGE( SEDEENSDK REQUIRED 
                HINTS ../../.. 
                    "$ENV{${PROGRAMFILESX86}}/Sedeen Viewer SDK/v5.4.1.20190327/msvc2017"
                    "$ENV{PROGRAMFILES}/Sedeen Viewer SDK/v5.4.1.20190327/msvc2017" )

# There is a dependency on boost.filesystem
IF(NOT BOOST_ROOT)
  SET(BOOST_ROOT "BOOST_ROOT-NOTFOUND" CACHE PATH "Installation prefix of the Boost C++ libraries")
ENDIF()
IF(NOT BOOST_VERSION)
  SET(BOOST_VERSION "BOOST_VERSION-NOTFOUND" CACHE STRING "Boost library version number")
ENDIF()
FIND_PACKAGE(Boost ${BOOST_VERSION} REQUIRED COMPONENTS
                 filesystem )

# Load the included OpenCV libs
#FIND_PACKAGE(SEDEENSDK_OPENCV REQUIRED
#             HINTS ../../..
#                  "$ENV{${PROGRAMFILESX86}}/Sedeen Viewer SDK/v5.4.1.20190327/msvc2017"
#                  "$ENV{PROGRAMFILES}/Sedeen Viewer SDK/v5.4.1.20190327/msvc2017" )

## Load OpenCV dependency
FIND_PACKAGE(OPENCV REQUIRED)
#SET (CMAKE_PREFIX_PATH "D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install" )
#FIND_PACKAGE( OPENCV REQUIRED "D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install" )

#INCLUDE_DIRECTORIES( "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/include" 
#		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include" 
#		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include/opencv" 
#		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/include/opencv2"
#		"C:/Boost/include/boost-1_55")

#LINK_DIRECTORIES( "C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/lib" 
#		"D:/Documents/Libraries/opencv-3.1.0_Contrib/build/install/x86/vc12/lib"
#		"C:/Boost/lib/i386" 
#		"C:/Program Files (x86)/Sedeen Viewer SDK/v5.2.3.629/msvc2012/bin")


INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${SEDEENSDK_INCLUDE_DIR} 
                     ${BOOST_ROOT} 
                     #${SEDEENSDK_OPENCV_INCLUDE_DIR} I need to get and compile opencv contrib
                     "D:/libraries/opencv/Install/include"
                     "D:/libraries/opencv/Install/include/opencv"
                     "D:/libraries/opencv/Install/include/opencv2"
                     )
LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SEDEENSDK_LIBRARY_DIR} 
                     "${BOOST_ROOT}/stage/lib" 
                     #${SEDEENSDK_OPENCV_LIBRARY_DIR} I need to get and compile opencv contrib
                     "D:/libraries/opencv/Install/x64/vc15/lib"
                     ) 

##
## Build the code into a module library
ADD_LIBRARY( ${PROJECT_NAME} MODULE ${PROJECT_NAME}.h ${PROJECT_NAME}.cpp 
                                    tinyxml2.h tinyxml2.cpp ) 

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${SEDEENSDK_LIBRARIES}
                           #${SEDEENSDK_OPENCV_LIBRARIES} 
                           Boost::filesystem 
                           image.lib 
                           global.lib 
                           algorithm.lib 
                           archive.lib 
                           bindings_opencv.lib 
                           opencv_core340.lib
                           opencv_imgproc340.lib
                           #opencv_ts340.lib 
                           opencv_features2d340.lib 
                           opencv_xfeatures2d340.lib
                           )

# Link the library against the Sedeen libraries
# NOTE: The QT libraries must be linked first.
#TARGET_LINK_LIBRARIES( RigidRegistration image.lib global.lib algorithm.lib archive.lib 
#		bindings_opencv.lib opencv_core310.lib opencv_imgproc310.lib opencv_ts310.lib 
#		opencv_features2d310.lib opencv_xfeatures2d310.lib) 

## 
##
## Install the plugin in the sedeen plugins directory
##IF( ${PATHCORE_FOUND} )
#  INSTALL( TARGETS RigidRegistration LIBRARY DESTINATION "C:/Program Files (x86)/Sedeen Viewer/plugins/cpp/piip/RigidRegistration" )
##ENDIF()


#Create or update the .info file in the build directory
STRING( TIMESTAMP DATE_CREATED_TEXT "%Y-%m-%d" )
CONFIGURE_FILE( "infoTemplate.info.in" "${PROJECT_NAME}.info" )

# Set the install destination directory
IF( NOT PLUGIN_DESTINATION_DIR )
  IF( ${SEDEEN_FOUND} )
	SET( TEMPPLUGINDIR "${PATHCORE_DIR}/plugins/cpp/piip/${PROJECT_NAME}" )
  ELSE()
	SET( TEMPPLUGINDIR "PLUGIN_DESTINATION_DIR-NOTFOUND" )
	MESSAGE( SEND_ERROR "PLUGIN_DESTINATION_DIR not found. Set this to the target installation directory of the plugin within Sedeen Viewer (e.g. $ENV{PROGRAMFILES}/Sedeen Viewer/plugins/cpp/piip/${PROJECT_NAME}).")
  ENDIF()
  SET(PLUGIN_DESTINATION_DIR ${TEMPPLUGINDIR} CACHE PATH "Installation directory for the plugin within Sedeen Viewer")
ENDIF()

# Install the plugin and .info file in the PLUGIN_DESTINATION_DIR directory
IF( ${SEDEEN_FOUND} )
  INSTALL(TARGETS ${PROJECT_NAME}
      LIBRARY DESTINATION "${PLUGIN_DESTINATION_DIR}")
  INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.info"
      DESTINATION "${PLUGIN_DESTINATION_DIR}")
  #INSTALL(DIRECTORY "${SEDEENSDK_DIR}/bin/" 
  #      DESTINATION "${PLUGIN_DESTINATION_DIR}" 
  #      FILES_MATCHING PATTERN "opencv*0.dll" )
  INSTALL(DIRECTORY "D:/libraries/opencv/Install/x64/vc15/bin/"
        DESTINATION "${PLUGIN_DESTINATION_DIR}" 
        FILES_MATCHING PATTERN "opencv*0.dll" )
ENDIF()

#Shows all variables and their values
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
