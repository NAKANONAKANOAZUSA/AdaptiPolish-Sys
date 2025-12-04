#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace py = pybind11;

// 读取 PLY → 返回 numpy (N,3)
py::array_t<float> load_pointcloud_cpp(const std::string &path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPLYFile(path, *cloud) < 0) {
        throw std::runtime_error("Failed to load PLY: " + path);
    }

    size_t N = cloud->points.size();
    py::array_t<float> arr({N, (size_t)3});
    auto buf = arr.mutable_unchecked<2>();

    for (size_t i = 0; i < N; i++) {
        buf(i, 0) = cloud->points[i].x;
        buf(i, 1) = cloud->points[i].y;
        buf(i, 2) = cloud->points[i].z;
    }

    return arr;
}

// xyz + labels → 写带颜色的 ASCII PLY
void save_colored_ply_cpp(const std::string &path,
                          py::array_t<float> xyz,
                          py::array_t<int> labels) {

    auto xyz_buf = xyz.unchecked<2>();
    auto lab_buf = labels.unchecked<1>();

    size_t N = xyz_buf.shape(0);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.resize(N);

    for (size_t i = 0; i < N; i++) {
        float x = xyz_buf(i, 0);
        float y = xyz_buf(i, 1);
        float z = xyz_buf(i, 2);
        int cls = lab_buf(i);

        pcl::PointXYZRGB p;
        p.x = x; p.y = y; p.z = z;

        // 颜色映射
        if (cls == 0) { p.r = 128; p.g = 128; p.b = 128; }      // 环境 灰
        else if (cls == 1) { p.r = 0; p.g = 0; p.b = 255; }     // 工件 蓝
        else if (cls == 2) { p.r = 255; p.g = 0; p.b = 0; }     // 瑕疵 红
        else { p.r = 0; p.g = 255; p.b = 0; }                   // 其他类 绿

        cloud->points[i] = p;
    }

    cloud->width = N;
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::io::savePLYFileASCII(path, *cloud);
}

// pybind11 导出模块
PYBIND11_MODULE(pc_backend, m) {
    m.doc() = "C++ point cloud backend using PCL";

    m.def("load_pointcloud", &load_pointcloud_cpp,
          "Load PLY file and return Nx3 float32 numpy array");

    m.def("save_colored_ply", &save_colored_ply_cpp,
          "Save colored ply from xyz and labels");
}
