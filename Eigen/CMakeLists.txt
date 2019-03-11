
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(RegexUtils)
test_escape_string_as_regex()

file(GLOB Eigen_directory_files "*")

escape_string_as_regex(ESCAPED_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

foreach(f ${Eigen_directory_files})
  if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/src")
    list(APPEND Eigen_directory_files_to_install ${f})
  endif()
endforeach(f ${Eigen_directory_files})

#message(${Eigen_directory_files_to_install})
add_library(Eigen ${Eigen_directory_files_to_install})

#install(FILES
#  ${Eigen_directory_files_to_install}
#  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel
#  )

#install(DIRECTORY src DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel FILES_MATCHING PATTERN "*.h")
