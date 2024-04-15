cmake_minimum_required(VERSION 3.0)
project(UAV)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/cuda-10.2/include /home/ubuntu/wjd/json/single_include/)
link_directories(/usr/local/cuda-10.2/lib64)

set(SRC src/main.cpp 
        src/infer.cu
        src/yolo.cu
        src/rotate.cpp
        )
add_executable(a ${SRC})

target_link_libraries(a nvinfer cudart ${OpenCV_LIBS})



# cmake_minimum_required(VERSION 3.0)
# project(UAV)
# find_package(OpenCV REQUIRED)
# include_directories(/home/ubuntu/wjd/json/single_include/ ${OpenCV_INCLUDE_DIRS})

# set(SRC src/main.cpp 
#         )
# add_executable(a ${SRC} )

# target_link_libraries(a  ${OpenCV_LIBS})
