#include "g2o_types.h"
#include "icp2d.h"

#include <execution>

void VertexSE2::oplusImpl(const double *update) {
  // strictly speaking optimiztion is not performed on SE2 manifold
  // instead it's on Euclidean space, i.e., a regular vector space
  // On SE2 manifold it should be update as
  // _estimate *= update
  Eigen::Map<const Eigen::Vector3d> delta(update);
  _estimate.translation() += delta.head<2>();
  _estimate.so2() = _estimate.so2() * Sophus::SO2d::exp(delta[2]);
}

void EdgeP2P::computeError() {
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

void EdgeP2P::linearizeOplus() {
  const double range = measurement()[0];
  const double angle = measurement()[1];
  const double vertex_angle =
      static_cast<VertexSE2 *>(_vertices[0])->estimate().so2().log();
  _jacobianOplusXi << 1, 0, -range * std::sin(angle + vertex_angle), 0, 1,
      range * std::cos(angle + vertex_angle);
}

void EdgeP2L::computeError() {
  const double range = measurement()[0];
  const double angle = measurement()[1];
  VertexSE2 *vertex = static_cast<VertexSE2 *>(_vertices[0]);

  const ICP2D::Point query_pt =
      vertex->estimate() * ICP2D::scan2point(range, angle);

  std::vector<int> nidx;
  std::vector<double> ndist;
  kdtree_->nearest_neighbors(query_pt, ICP2D::kmax_p2l_num, nidx, ndist);

  std::vector<ICP2D::Point> points;
  points.reserve(ICP2D::kmax_p2l_num);
  for (size_t nid = 0; nid < nidx.size(); ++nid) {
    if (nidx[nid] == kno_match || ndist[nid] > 3 * ICP2D::kmax_p2p_dist2) {
      continue;
    }
    points.emplace_back(kdtree_->get_point(nidx[nid]));
  }

  valid_fitting_ = ICP2D::line_fitting(points, line_coeffs_);
  if (!valid_fitting_) {
    _error << 0;
    return;
  }
  _error << line_coeffs_[0] * query_pt.x() + line_coeffs_[1] * query_pt.y() +
                line_coeffs_[2];
}

void EdgeP2L::linearizeOplus() {
  if (!valid_fitting_) {
    _jacobianOplusXi << 0, 0, 0;
    return;
  }
  const double range = measurement()[0];
  const double angle = measurement()[1];
  const double theta =
      static_cast<VertexSE2 *>(_vertices[0])->estimate().so2().log();

  _jacobianOplusXi << line_coeffs_[0], line_coeffs_[1],
      -line_coeffs_[0] * range * std::sin(theta + angle) +
          line_coeffs_[1] * range * std::cos(theta + angle);
}

void EdgeP2LMT::computeError() {
  VertexSE2 *vertex = static_cast<VertexSE2 *>(_vertices[0]);
  ICP2D::PointCloud query_cloud(ranges_.size(), Eigen::Vector2d::Zero());
  KDTree<double, 2> *kdtree = measurement();
  std::vector<int> idx(ranges_.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [this, &vertex, &query_cloud](const int qid) {
                  query_cloud[qid] =
                      vertex->estimate() *
                      ICP2D::scan2point(ranges_[qid], angles_[qid]);
                });
  std::vector<std::vector<int>> nnidx;
  std::vector<std::vector<double>> nndist;
  kdtree->nearest_neighbors_kmt(query_cloud, ICP2D::kmax_p2l_num, nnidx,
                                nndist);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [this, &query_cloud, &kdtree, &nnidx, &nndist](const int qid) {
                  const std::vector<int> &nidx = nnidx[qid];
                  const std::vector<double> &ndist = nndist[qid];

                  std::vector<ICP2D::Point> points;
                  points.reserve(nidx.size());
                  for (size_t nid = 0; nid < nidx.size(); ++nid) {
                    if (nidx[nid] == kno_match ||
                        ndist[nid] > 3 * ICP2D::kmax_p2p_dist2) {
                      continue;
                    }
                    points.emplace_back(kdtree->get_point(nidx[nid]));
                  }

                  line_coeffs_[qid].setZero();
                  valid_fittings_[qid] =
                      ICP2D::line_fitting(points, line_coeffs_[qid]);
                  if (!valid_fittings_[qid]) {
                    _error[qid] = 0;
                  } else {
                    _error[qid] = line_coeffs_[qid][0] * query_cloud[qid].x() +
                                  line_coeffs_[qid][1] * query_cloud[qid].y() +
                                  line_coeffs_[qid][2];
                  }
                });
}

void EdgeP2LMT::linearizeOplus() {
  std::vector<int> idx(ranges_.size());
  std::iota(idx.begin(), idx.end(), 0);
  const double theta =
      static_cast<VertexSE2 *>(_vertices[0])->estimate().so2().log();
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [this, &theta](const int qid) {
                  if (valid_fittings_[qid]) {
                    _jacobianOplusXi.row(qid) << line_coeffs_[qid][0],
                        line_coeffs_[qid][1],
                        -line_coeffs_[qid][0] * ranges_[qid] *
                                std::sin(angles_[qid] + theta) +
                            line_coeffs_[qid][1] * ranges_[qid] *
                                std::cos(angles_[qid] + theta);
                  } else {
                    _jacobianOplusXi.row(qid).setZero();
                  }
                });
}

void EdgeLikelihood::computeError() {
  VertexSE2 *vertex = static_cast<VertexSE2 *>(_vertices[0]);
  const Eigen::Vector2d query_pos = vertex->estimate() * measurement();
  const Eigen::Vector2d query_coord =
      query_pos * likelihood_field_->resolution() +
      Eigen::Vector2d(likelihood_field_->img_offset(),
                      likelihood_field_->img_offset()) -
      Eigen::Vector2d(0.5, 0.5);

  if (!likelihood_field_->outside(query_coord.x(), query_coord.y(), 10)) {

    _error[0] = likelihood_field_->get_value(query_coord.y(), query_coord.x());
  } else {
    _error[0] = 0;
    setLevel(1);
  }
}
void EdgeLikelihood::linearizeOplus() {
  VertexSE2 *vertex = static_cast<VertexSE2 *>(_vertices[0]);
  const double theta = vertex->estimate().so2().log();
  const Eigen::Vector2d query_pos = vertex->estimate() * measurement();
  const Eigen::Vector2d query_coord =
      query_pos * likelihood_field_->resolution() +
      Eigen::Vector2d(likelihood_field_->img_offset(),
                      likelihood_field_->img_offset()) -
      Eigen::Vector2d(0.5, 0.5);

  if (!likelihood_field_->outside(query_coord.x(), query_coord.y(), 10)) {

    double grad_x =
        0.5 *
        (likelihood_field_->get_value(query_coord.y(), query_coord.x() + 1) -
         likelihood_field_->get_value(query_coord.y(), query_coord.x() - 1));
    double grad_y =
        0.5 *
        (likelihood_field_->get_value(query_coord.y() + 1, query_coord.x()) -
         likelihood_field_->get_value(query_coord.y() - 1, query_coord.x()));

    _jacobianOplusXi << likelihood_field_->resolution() * grad_x,
        likelihood_field_->resolution() * grad_y,
        likelihood_field_->resolution() * range_ *
            (-grad_x * std::sin(angle_ + theta) +
             grad_y * std::cos(angle_ + theta));

  } else {
    _jacobianOplusXi.setZero();
    setLevel(1);
  }
}

void EdgeSubmaps::computeError() {
  VertexSE2 *vertex0 = static_cast<VertexSE2 *>(_vertices[0]);
  VertexSE2 *vertex1 = static_cast<VertexSE2 *>(_vertices[1]);
  _error = (vertex0->estimate().inverse() * vertex1->estimate() *
            measurement().inverse())
               .log();
}