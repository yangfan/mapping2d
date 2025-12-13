#include "g2o_types.h"
#include "icp2d.h"

void VertexSE2::oplusImpl(const double *update) {
  Eigen::Map<const Eigen::Vector3d> delta(update);
  _estimate.translation() += delta.head<2>();
  _estimate.so2() = _estimate.so2() * Sophus::SO2d::exp(delta[2]);
}

void EdgeScan2D::computeError() {
  VertexSE2 *vertex = static_cast<VertexSE2 *>(_vertices[0]);
  ICP2D::Point query_pt = vertex->estimate() *
                          ICP2D::scan2point(measurement()[0], measurement()[1]);
  std::vector<int> nidx;
  std::vector<double> ndist;
  if (!kdtree_->nearest_neighbors(query_pt, 1, nidx, ndist)) {
    _error.setZero();
    return;
  }
  _error = query_pt - kdtree_->get_point(nidx[0]);
}

void EdgeScan2D::linearizeOplus() {
  const double range = measurement()[0];
  const double angle = measurement()[1];
  const double vertex_angle =
      static_cast<VertexSE2 *>(_vertices[0])->estimate().so2().log();
  _jacobianOplusXi << 1, 0, -range * std::sin(angle + vertex_angle), 0, 1,
      range * std::cos(angle + vertex_angle);
}