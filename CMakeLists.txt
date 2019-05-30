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

# There is a dependency on Boost
#IF(NOT BOOST_ROOT)
#  SET(BOOST_ROOT "BOOST_ROOT-NOTFOUND" CACHE PATH "Installation prefix of the Boost C++ libraries")
#ENDIF()
#IF(NOT BOOST_VERSION)
#  SET(BOOST_VERSION "BOOST_VERSION-NOTFOUND" CACHE STRING "Boost library version number")
#ENDIF()
#FIND_PACKAGE(Boost ${BOOST_VERSION} REQUIRED ) 

## Load OpenCV dependency, set value of OPENCV_DIR
FIND_PACKAGE(OPENCV REQUIRED)
SET(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/include" CACHE PATH "Directory containing OpenCV include files, has opencv and opencv2 subdirectories")
SET(OPENCV_LIB_DIR "${OPENCV_DIR}/x64/vc15/lib" CACHE PATH "Directory containing static OpenCV libraries")
SET(OPENCV_DLL_DIR "${OPENCV_DIR}/x64/vc15/bin" CACHE PATH "Directory containing dynamic OpenCV libraries")

INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${SEDEENSDK_INCLUDE_DIR} 
                     ${BOOST_ROOT} 
                     ${OPENCV_INCLUDE_DIR}
                     "${OPENCV_INCLUDE_DIR}/opencv"
                     "${OPENCV_INCLUDE_DIR}/opencv2"
                     )
LINK_DIRECTORIES( ${LINK_DIRECTORIES} ${SEDEENSDK_LIBRARY_DIR} 
                     "${BOOST_ROOT}/stage/lib" 
                     ${OPENCV_LIB_DIR}
                     ) 

##
## Build the code into a module library
ADD_LIBRARY( ${PROJECT_NAME} MODULE ${PROJECT_NAME}.h ${PROJECT_NAME}.cpp 
                                    tinyxml2.h tinyxml2.cpp ) 

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${SEDEENSDK_LIBRARIES}
                           bindings_opencv.lib 
                           opencv_core340.lib
                           opencv_imgproc340.lib
                           opencv_features2d340.lib 
                           opencv_xfeatures2d340.lib
                           )

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

  INSTALL(DIRECTORY "${OPENCV_DLL_DIR}/"
        DESTINATION "${PLUGIN_DESTINATION_DIR}" 
        FILES_MATCHING PATTERN "opencv*0.dll" )
ENDIF()

#Shows all variables and their values
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
