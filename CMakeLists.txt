cmake_minimum_required(VERSION 3.2.2)

project(SYCL-AccessorsNote LANGUAGES CXX)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp)

foreach(Src IN ITEMS Minimal MinimalFix Normalize)
  add_executable(${Src} ${Src}.cpp)

  set_target_properties(${Src} PROPERTIES CXX_STANDARD 17
                                          CXX_STANDARD_REQUIRED ON
                                          CXX_EXTENSIONS OFF)

  target_compile_options(${Src} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)
  add_sycl_to_target(TARGET ${Src}
                     SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${Src}.cpp)
endforeach()