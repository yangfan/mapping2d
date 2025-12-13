#pragma once

#include "KDTree.hpp"

#include <Eigen/Core>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <sophus/se2.hpp>

#include <iostream>

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

class EdgeScan2D : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSE2> {
public:
  EdgeScan2D(KDTree<double, 2> *kdtree) : kdtree_(kdtree) {}
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