#pragma once
#include <ceres/ceres.h>
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
      : likelihood_field(lf), query_pt_s_(query_pt_s), range_(range),
        angle_(angle) {}
  bool Evaluate(double const *const *params, double *residuals,
                double **jacobian) const override;

private:
  LikelihoodField *likelihood_field = nullptr;
  Eigen::Vector2d query_pt_s_ = Eigen::Vector2d::Zero();
  double range_ = 0;
  double angle_ = 0;
};