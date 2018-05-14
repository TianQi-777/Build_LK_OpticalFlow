# Build_LK_OpticalFlow
The purpose of this demo is to better understand the underlying implementation of single level optical flow and multi level optical flow with forward or inverse.

## Related theory
[Optical flow](https://en.wikipedia.org/wiki/Optical_flow) is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.
You can read more about the optical flow method by reading [Lucas-Kanade 20 Years On: A Unifying Framework](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf)

## Additional Prerequisites for this demo  
Besides,to build this project, you need the followings:  

**OpenCV**  
Use [OpenCV](http://opencv.org) to process images.

**Eigen3**  
Download and install instructions can be found at: http://eigen.tuxfamily.org. 

**C++11 or C++0x Compiler**  
Use the some functionalities of C++11.

## Build and Run
```
cd XX/XX(include optical_flow.cpp ,1.png , 2.png and CMakeLists.txt)  
mkdir build  
cd build  
cmake ..  
make -j2  
./optical_flow
```

## Result
<div align=center>  
  
![](https://github.com/TianQi-777/Build_LK_OpticalFlow/blob/master/Images/opencv.png)
</div>

<div align=center>  
  
![](https://github.com/TianQi-777/Build_LK_OpticalFlow/blob/master/Images/LK.png)
</div>
