#pragma once
#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se2.hpp>

#include <vector>

#include "KDTree.hpp"
#include "LikelihoodField.h"

class ICP2DCeresP2P : public ceres::SizedCostFunction<2, 3> {
public:
  ICP2DCeresP2P(const double range, const double angle,
                KDTree<double, 2> *kdtree)
      : range_(range), angle_(angle), kdtree_(kdtree) {}

  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override;

private:
  double range_ = 0.0;
  double angle_ = 0.0;
  KDTree<double, 2> *kdtree_;
};

class ICP2DCeresP2L : public ceres::SizedCostFunction<1, 3> {
public:
  ICP2DCeresP2L(const double range, const double angle,
                KDTree<double, 2> *kdtree)
      : range_(range), angle_(angle), kdtree_(kdtree) {}

  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override;

private:
  double range_ = 0.0;
  double angle_ = 0.0;
  KDTree<double, 2> *kdtree_;
};

class ICP2DCeresP2LMT : public ceres::SizedCostFunction<ceres::DYNAMIC, 3> {
public:
  ICP2DCeresP2LMT(std::vector<double> &&ranges, std::vector<double> &&angles,
                  KDTree<double, 2> *kdtree)
      : ranges_(std::move(ranges)), angles_(std::move(angles)),
        kdtree_(kdtree) {
    set_num_residuals(ranges_.size());
  }
  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override;

private:
  std::vector<double> ranges_;
  std::vector<double> angles_;
  KDTree<double, 2> *kdtree_;
};

class LikelihoodAlignment : public ceres::SizedCostFunction<1, 3> {
public:
  LikelihoodAlignment(LikelihoodField *lf, const Eigen::Vector2d &query_pt_s,
                      const double range, const double angle)
      : likelihood_field_(lf), query_pt_s_(query_pt_s), range_(range),
        angle_(angle), is_outlier_(new bool(false)) {}
  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override;
  bool is_outlier() const { return *is_outlier_; }

private:
  LikelihoodField *likelihood_field_ = nullptr;
  Eigen::Vector2d query_pt_s_ = Eigen::Vector2d::Zero();
  double range_ = 0;
  double angle_ = 0;
  std::unique_ptr<bool> is_outlier_;
};

struct SE2EdgeCost {
  SE2EdgeCost(const Sophus::SE2d &m, const Eigen::Matrix3d &sqrt_info_mat)
      : T12(m), sqrt_info(sqrt_info_mat) {}
  template <typename T>
  bool operator()(const T *const data1, const T *const data2,
                  T *residual) const {
    Eigen::Map<const Sophus::SE2<T>> vertex1(data1);
    Eigen::Map<const Sophus::SE2<T>> vertex2(data2);
    Eigen::Map<typename Sophus::SE2<T>::Tangent> err(residual);
    err =
        sqrt_info.template cast<T>() *
        (vertex1.inverse() * vertex2 * T12.inverse().template cast<T>()).log();
    return true;
  }

  Sophus::SE2d T12;
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
};