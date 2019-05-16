#include <Python.h>
#include <apriltag_tracker/AprilTagTracker.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <memory>

template <typename T>
static inline Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
toEigenMap(
    PyObject *obj) {
  auto array_obj = reinterpret_cast<PyArrayObject *>(obj);
  if (!array_obj)
    throw std::runtime_error("ptr is not PyArrayObject");

  int ndims = PyArray_NDIM(array_obj);
  if (ndims != 2) {
    throw std::runtime_error(
        "expecting a 2D array but got " + std::to_string(ndims));
  }

  auto size_ptr = PyArray_DIMS(array_obj);
  auto rows = size_ptr[0], cols = size_ptr[1];

  if (0 == rows || 0 == cols)
    throw std::runtime_error("empty array");

  auto ptr = reinterpret_cast<T *>(PyArray_DATA(array_obj));
  if (nullptr == ptr)
    throw std::runtime_error("failed to get the right type");

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
  Eigen::Map<Matrix> ret(ptr, rows, cols);

  return ret;
}

class AprilTagTrackerPythonWrapper {
 private:
  std::shared_ptr<AprilTagTracker> tracker_;

 public:
  AprilTagTrackerPythonWrapper(std::string params_path) {
    tracker_ = std::make_shared<AprilTagTracker>(params_path);
    std::cout << "[AprilTagTrackerPythonWrapper] Initialization done!\n";
  }

  bool PythonTrack(PyObject *curr_img_py, PyObject *theta_py) {
    auto curr_obj = reinterpret_cast<PyArrayObject *>(curr_img_py);
    int ndims = PyArray_NDIM(curr_obj);
    if (ndims != 2) {
      throw std::runtime_error(
          "expecting a 2D array but got " + std::to_string(ndims));
    }
    auto size_ptr = PyArray_DIMS(curr_obj);
    auto rows = size_ptr[0], cols = size_ptr[1];

    auto curr_ptr = PyArray_DATA(reinterpret_cast<PyArrayObject *>(curr_img_py));
    cv::Mat curr(cv::Size(cols, rows), CV_8UC1, static_cast<uchar *>(curr_ptr));
    auto theta = toEigenMap<float>(theta_py);

    cv::Mat debug_img;
    if (tracker_->TrackTag(curr,
                           theta,
                           debug_img)) {
      return true;
    }
    return false;
  }

  bool PythonTrackDebug(PyObject *curr_img_py, PyObject *theta_py, PyObject *debug_img_py) {
    auto curr_obj = reinterpret_cast<PyArrayObject *>(curr_img_py);
    int ndims = PyArray_NDIM(curr_obj);
    if (ndims != 2) {
      throw std::runtime_error(
          "expecting a 2D array but got " + std::to_string(ndims));
    }
    auto size_ptr = PyArray_DIMS(curr_obj);
    auto rows = size_ptr[0], cols = size_ptr[1];

    auto curr_ptr = PyArray_DATA(reinterpret_cast<PyArrayObject *>(curr_img_py));
    cv::Mat curr(cv::Size(cols, rows), CV_8UC1, static_cast<uchar *>(curr_ptr));
    auto theta = toEigenMap<float>(theta_py);

    auto debug_ptr = PyArray_DATA(reinterpret_cast<PyArrayObject *>(debug_img_py));
    cv::Mat debug_img(cv::Size(cols, rows), CV_8UC3, static_cast<uchar *>(debug_ptr));

    if (tracker_->TrackTag(curr,
                           theta,
                           debug_img)) {
      return true;
    }
    return false;
  }

  boost::python::list PythonDetectTagDebug(PyObject *curr_img_py,
                                           PyObject *debug_img_py) {
    auto curr_obj = reinterpret_cast<PyArrayObject *>(curr_img_py);
    int ndims = PyArray_NDIM(curr_obj);
    if (ndims != 2) {
      throw std::runtime_error("expecting a 2D array but got " +
                               std::to_string(ndims));
    }
    auto size_ptr = PyArray_DIMS(curr_obj);
    auto rows = size_ptr[0], cols = size_ptr[1];

    auto curr_ptr =
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(curr_img_py));
    cv::Mat curr(cv::Size(cols, rows), CV_8UC1, static_cast<uchar *>(curr_ptr));

    std::vector<cv::Point2f> detections;

    auto debug_ptr =
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(debug_img_py));
    cv::Mat debug_img(cv::Size(cols, rows), CV_8UC3,
                      static_cast<uchar *>(debug_ptr));

    boost::python::list list;
    if (tracker_->DetectTag(curr, detections, debug_img)) {
      // Copy over the detections to the pyobject
      for (auto &pixel : detections) {
        list.append(pixel.x);
        list.append(pixel.y);
      }
    }
    return list;
  }
};

using namespace boost::python;
BOOST_PYTHON_MODULE(_AprilTagTracker) {
  class_<AprilTagTrackerPythonWrapper>("AprilTagTracker", init<std::string>())
      .def("track", &AprilTagTrackerPythonWrapper::PythonTrack)
      .def("track_debug", &AprilTagTrackerPythonWrapper::PythonTrackDebug)
      .def("detect_tag", &AprilTagTrackerPythonWrapper::PythonDetectTagDebug);
}
