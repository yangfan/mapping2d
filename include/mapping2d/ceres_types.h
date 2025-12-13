#pragma once
#include <ceres/ceres.h>
#include <sophus/se2.hpp>

#include <vector>

#include "KDTree.hpp"
#include "icp2d.h"

// class ICP2dCeresAuto {
// public:
//   ICP2dCeresAuto(const double b_x, const double b_y, KDTree<double, 2>
//   *kdtree)
//       : b_x_(b_x), b_y_(b_y), kdtree_(kdtree) {}
//   template <typename T>
//   bool operator()(const T *const pose, T *residual) const {
//     Sophus::SE2<T> T_wb =
//         Sophus::SE2<T>(pose[2], Eigen::Matrix<T, 2, 1>(pose[0], pose[1]));
//     Eigen::Matrix<T, 2, 1> query_pt = T_wb * Eigen::Matrix<T, 2, 1>(b_x_,
//     b_y_); std::vector<int> nidx; std::vector<double> ndist;
//     kdtree_->nearest_neighbors(query_pt, 1, nidx, ndist);
//     Eigen::Matrix<T, 2, 1> err =
//         query_pt - kdtree_->get_point(nidx[0]).cast<T>();
//     residual[0] = err[0];
//     residual[1] = err[1];

//     return true;
//   }
//   static ceres::CostFunction *create(const double range, const double angle,
//                                      KDTree<double, 2> *kdtree) {
//     return (new ceres::AutoDiffCostFunction<ICP2dCeresAuto, 2, 3>(
//         new ICP2dCeresAuto(range, angle, kdtree)));
//   }

// private:
//   // double range_ = 0.0;
//   // double angle_ = 0.0;
//   double b_x_ = 0.0;
//   double b_y_ = 0.0;
//   KDTree<double, 2> *kdtree_ = nullptr;
// };

class ICP2DCeres : public ceres::SizedCostFunction<2, 3> {
public:
  // ICP2DCeres(const ICP2D::Point &source_pt, KDTree<double, 2> *kdtree)
  //     : source_pt_(source_pt), kdtree_(kdtree) {}
  ICP2DCeres(const double range, const double angle, KDTree<double, 2> *kdtree)
      : range_(range), angle_(angle), kdtree_(kdtree) {}

  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override {
    Sophus::SE2d pose(params[0][2],
                      Eigen::Vector2d(params[0][0], params[0][1]));
    // ICP2D::Point query_pt = pose * source_pt_;
    ICP2D::Point query_pt = pose * ICP2D::scan2point(range_, angle_);
    std::vector<int> nidx;
    std::vector<double> ndist;
    if (kdtree_->nearest_neighbors(query_pt, 1, nidx, ndist)) {
      ICP2D::Point err = query_pt - kdtree_->get_point(nidx[0]);
      residuals[0] = err.x();
      residuals[1] = err.y();
    } else {
      residuals[0] = 0;
      residuals[1] = 0;
    }

    if (jacobian && jacobian[0]) {
      jacobian[0][0] = 1;
      jacobian[0][1] = 0;
      jacobian[0][2] = -range_ * std::sin(angle_ + pose.so2().log());
      jacobian[0][3] = 0;
      jacobian[0][4] = 1;
      jacobian[0][5] = range_ * std::cos(angle_ + pose.so2().log());
    }
    return true;
  }

private:
  double range_ = 0.0;
  double angle_ = 0.0;
  // ICP2D::Point source_pt_;
  KDTree<double, 2> *kdtree_;
};
