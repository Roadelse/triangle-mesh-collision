# triangle-mesh-collision

Self collision detection for triangles meshes. Implementation in C++, uses Eigen and [libigl](http://libigl.github.io/libigl/) libraries.


UPDATE:

+ Use `-DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmak` to cmake
+ Change algorithm from counting collisions to having any collision or not
+ Provide an alternative python version, slower but more flexible



**To use:**  
  1. Create build directory (`mkdir build`)  
  2. `cd build/`  
  3. Run cmake on project (`cmake ..`)  
  4. Make project (`make`)  
  5. Place object file into `meshes/` directory. 
  6. Inside build folder, run `./collision_bin FILE_NAME.EXT`

**Notes**

* Currently only supports .off and .ply file types. Check the available `igl::readEXT()` methods to add support for different file types. Add to `getTriangleMesh()` method in `main.cpp`
* Toggle the bool `VISUALIZATION` at top of `main.cpp` to show/hide object and collision results
* Make sure CMake can find libigl on your machine. Inside the `cmake/FindLIBIGL.cmake`, include the path to your local copy of libigl
