#pragma once

#include "KDTree.hpp"
#include "LikelihoodField.h"

#include <Eigen/Core>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <sophus/se2.hpp>

#include <iostream>
#include <vector>

class VertexSE2 : public g2o::BaseVertex<3, Sophus::SE2d> {
public:
  VertexSE2() {}
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void setToOriginImpl() override { _estimate = Sophus::SE2d(); }
  virtual void oplusImpl(const double *update) override;
};

class EdgeP2P : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE2> {
public:
  EdgeP2P(KDTree<double, 2> *kdtree) : kdtree_(kdtree) {}
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void computeError() override;
  virtual void linearizeOplus() override;

private:
  KDTree<double, 2> *kdtree_ = nullptr;
};

class EdgeP2L : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexSE2> {
public:
  EdgeP2L(KDTree<double, 2> *kd) : kdtree_(kd) {}
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void computeError() override;
  virtual void linearizeOplus() override;

private:
  Eigen::Vector3d line_coeffs_ = Eigen::Vector3d::Zero();
  bool valid_fitting_ = false;
  KDTree<double, 2> *kdtree_ = nullptr;
};

class EdgeP2LMT
    : public g2o::BaseUnaryEdge<-1, KDTree<double, 2> *, VertexSE2> {
public:
  EdgeP2LMT(std::vector<double> &&ranges, std::vector<double> &&angles)
      : ranges_(std::move(ranges)), angles_(std::move(angles)) {
    setDimension(ranges_.size());
    line_coeffs_ =
        std::vector<Eigen::Vector3d>(ranges_.size(), Eigen::Vector3d::Zero());
    valid_fittings_ = std::vector<bool>(ranges_.size(), false);
  }
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void computeError() override;
  virtual void linearizeOplus() override;

private:
  std::vector<double> ranges_;
  std::vector<double> angles_;
  std::vector<Eigen::Vector3d> line_coeffs_;
  std::vector<bool> valid_fittings_;
};

class EdgeLikelihood
    : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexSE2> {
public:
  EdgeLikelihood(LikelihoodField *lf, const double range, const double angle)
      : likelihood_field_(lf), range_(range), angle_(angle) {}
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void computeError() override;
  virtual void linearizeOplus() override;

private:
  LikelihoodField *likelihood_field_ = nullptr;
  double range_ = 0.0;
  double angle_ = 0.0;
};

class EdgeSubmaps
    : public g2o::BaseBinaryEdge<3, Sophus::SE2d, VertexSE2, VertexSE2> {
public:
  virtual bool read(std::istream &ifs) override {
    ifs.clear();
    return false;
  }
  virtual bool write(std::ostream &ofs) const override {
    ofs.clear();
    return false;
  }
  virtual void computeError() override;
  // virtual void linearizeOplus() override;
};