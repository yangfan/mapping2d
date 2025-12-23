#include "ceres_types.h"
#include "icp2d.h"

bool ICP2DCeresP2P::Evaluate(double const *const *params, double *residuals,
                             double **jacobian) const {
  Sophus::SE2d pose(params[0][2], Eigen::Vector2d(params[0][0], params[0][1]));
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

bool ICP2DCeresP2L::Evaluate(double const *const *params, double *residuals,
                             double **jacobian) const {

  Sophus::SE2d pose(params[0][2], Eigen::Vector2d(params[0][0], params[0][1]));

  const ICP2D::Point query_pt = pose * ICP2D::scan2point(range_, angle_);

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

  Eigen::Vector3d line_coeffs = Eigen::Vector3d::Zero();
  if (!ICP2D::line_fitting(points, line_coeffs)) {
    residuals[0] = 0;
  } else {
    residuals[0] = line_coeffs[0] * query_pt.x() +
                   line_coeffs[1] * query_pt.y() + line_coeffs[2];
  }

  if (jacobian && jacobian[0]) {
    const double theta = pose.so2().log();
    jacobian[0][0] = line_coeffs[0];
    jacobian[0][1] = line_coeffs[1];
    jacobian[0][2] = -line_coeffs[0] * range_ * std::sin(theta + angle_) +
                     line_coeffs[1] * range_ * std::cos(theta + angle_);
  }
  return true;
}

// param: param_num X param_dim
// residuals: res_dim
// jacobian: param_num X (param_dim x res_dim)
// i.e., [param_0, param_i, param_]
bool ICP2DCeresP2LMT::Evaluate(double const *const *params, double *residuals,
                               double **jacobian) const {

  Sophus::SE2d pose(params[0][2], Eigen::Vector2d(params[0][0], params[0][1]));
  const double theta = pose.so2().log();
  const int N = ranges_.size();
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  ICP2D::PointCloud query_cloud(N);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [this, &query_cloud, &pose](const int qid) {
                  query_cloud[qid] =
                      pose * ICP2D::scan2point(ranges_[qid], angles_[qid]);
                });
  std::vector<std::vector<int>> nnidx;
  std::vector<std::vector<double>> nndist;
  kdtree_->nearest_neighbors_kmt(query_cloud, ICP2D::kmax_p2l_num, nnidx,
                                 nndist);
  std::for_each(
      std::execution::par_unseq, idx.begin(), idx.end(),
      [this, residuals, jacobian, &query_cloud, &nnidx, &nndist,
       &theta](const int qid) {
        const std::vector<int> &nidx = nnidx[qid];
        const std::vector<double> &ndist = nndist[qid];

        std::vector<ICP2D::Point> points;
        points.reserve(nidx.size());
        for (size_t nid = 0; nid < nidx.size(); ++nid) {
          if (nidx[nid] == kno_match ||
              ndist[nid] > 3 * ICP2D::kmax_p2p_dist2) {
            continue;
          }
          points.emplace_back(kdtree_->get_point(nidx[nid]));
        }

        Eigen::Vector3d line_coeffs = Eigen::Vector3d::Zero();
        if (!ICP2D::line_fitting(points, line_coeffs)) {
          residuals[qid] = 0;
        } else {
          residuals[qid] = line_coeffs[0] * query_cloud[qid].x() +
                           line_coeffs[1] * query_cloud[qid].y() +
                           line_coeffs[2];
        }
        if (jacobian && jacobian[0]) {
          jacobian[0][qid * 3] = line_coeffs[0];
          jacobian[0][qid * 3 + 1] = line_coeffs[1];
          jacobian[0][qid * 3 + 2] =
              -line_coeffs[0] * ranges_[qid] * std::sin(angles_[qid] + theta) +
              line_coeffs[1] * ranges_[qid] * std::cos(angles_[qid] + theta);
        }
      });
  return true;
}

bool LikelihoodAlignment::Evaluate(double const *const *params,
                                   double *residuals, double **jacobian) const {
  Sophus::SE2d pose(params[0][2], Eigen::Vector2d(params[0][0], params[0][1]));
  const Eigen::Vector2d query_coord =
      likelihood_field->resolution() * (pose * query_pt_s_) +
      likelihood_field->img_offset() * Eigen::Vector2d::Ones();

  const bool outside_map =
      likelihood_field->outside(query_coord.x(), query_coord.y());
  if (outside_map) {
    residuals[0] = 0;
  } else {
    residuals[0] =
        likelihood_field->get_value(query_coord.y(), query_coord.x());
  }

  if (jacobian && jacobian[0]) {
    if (outside_map) {
      jacobian[0][0] = 0;
      jacobian[0][1] = 0;
      jacobian[0][2] = 0;
    } else {
      const double theta = pose.so2().log();
      double grad_x =
          0.5 *
          (likelihood_field->get_value(query_coord.y(), query_coord.x() + 1) -
           likelihood_field->get_value(query_coord.y(), query_coord.x() - 1));
      double grad_y =
          0.5 *
          (likelihood_field->get_value(query_coord.y() + 1, query_coord.x()) -
           likelihood_field->get_value(query_coord.y() - 1, query_coord.x()));
      jacobian[0][0] = likelihood_field->resolution() * grad_x;
      jacobian[0][1] = likelihood_field->resolution() * grad_y;
      jacobian[0][2] = likelihood_field->resolution() * range_ *
                       (-grad_x * std::sin(angle_ + theta) +
                        grad_y * std::cos(angle_ + theta));
    }
  }
  return true;
}