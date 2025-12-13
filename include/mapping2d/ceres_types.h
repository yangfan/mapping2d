#pragma once
#include <ceres/ceres.h>
#include <sophus/se2.hpp>

#include <vector>

#include "KDTree.hpp"
#include "icp2d.h"

class ICP2DCeres : public ceres::SizedCostFunction<2, 3> {
public:
  ICP2DCeres(const double range, const double angle, KDTree<double, 2> *kdtree)
      : range_(range), angle_(angle), kdtree_(kdtree) {}

  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override {
    Sophus::SE2d pose(params[0][2],
                      Eigen::Vector2d(params[0][0], params[0][1]));
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
  KDTree<double, 2> *kdtree_;
};
