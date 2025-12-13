#include "KDTree.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <chrono>
#include <string>

DEFINE_string(pcd_path1, "./data/input/first.pcd", "pcd file 1");
DEFINE_string(pcd_path2, "./data/input/second.pcd", "pcd file 2");

template <typename Func>
void call_evaluate(Func &&func, const std::string &func_name,
                   const size_t iterations) {
  double elapsed = 0;
  for (size_t i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();
    elapsed +=
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
  }
  LOG(INFO) << func_name << " takes average " << elapsed / iterations << "ms.";
};

TEST(KDTest, BasicBuild3d) {
  using PointCloud = KDTree<double, 3>::PointCloud;
  using PointType = KDTree<double, 3>::PointType;
  PointType pt1, pt2, pt3, pt4;
  pt1 << 0, 0, 0;
  pt2 << 1, 0, 0;
  pt3 << 0, 1, 0;
  pt4 << 1, 1, 0;

  PointCloud cloud;
  cloud.push_back(pt1);
  cloud.push_back(pt2);
  cloud.push_back(pt3);
  cloud.push_back(pt4);

  KDTree<double, 3> kd_tree;
  kd_tree.setInputCloud(cloud);

  EXPECT_EQ(kd_tree.node_size(), 7);
  EXPECT_EQ(kd_tree.leaf_size(), cloud.size());
}

TEST(KDTest, BasicBuild2d) {
  using PointCloud = KDTree<double, 2>::PointCloud;
  using PointType = KDTree<double, 2>::PointType;
  PointType pt1, pt2, pt3, pt4;
  pt1 << 0, 0;
  pt2 << 1, 0;
  pt3 << 0, 1;
  pt4 << 1, 1;

  PointCloud cloud;
  cloud.push_back(pt1);
  cloud.push_back(pt2);
  cloud.push_back(pt3);
  cloud.push_back(pt4);

  KDTree<double, 2> kd_tree;
  kd_tree.setInputCloud(cloud);

  EXPECT_EQ(kd_tree.node_size(), 7);
  EXPECT_EQ(kd_tree.leaf_size(), cloud.size());
}

TEST(KDTest, kNN3D) {
  using PointCloud = KDTree<double, 3>::PointCloud;
  using pcl_PointType = pcl::PointXYZI;
  using pcl_PointCloud = pcl::PointCloud<pcl_PointType>;

  pcl_PointCloud::Ptr cloud1(new pcl_PointCloud);
  pcl_PointCloud::Ptr cloud2(new pcl_PointCloud);
  pcl::io::loadPCDFile(FLAGS_pcd_path1, *cloud1);
  pcl::io::loadPCDFile(FLAGS_pcd_path2, *cloud2);
  LOG(INFO) << "Number of point in cloud1: " << cloud1->size();
  LOG(INFO) << "Number of point in cloud2: " << cloud2->size();

  pcl::search::KdTree<pcl_PointType> kdtree_pcl;
  auto func_pcl_build = [&kdtree_pcl, &cloud1]() {
    kdtree_pcl.setInputCloud(cloud1);
  };
  call_evaluate(func_pcl_build, "func_pcl_build", 5);

  PointCloud pc1, pc2;
  pc1.reserve(cloud1->size());
  pc2.reserve(cloud2->size());
  std::for_each(cloud1->points.begin(), cloud1->points.end(),
                [&pc1](const pcl_PointType &pt) {
                  pc1.emplace_back(pt.getVector3fMap().cast<double>());
                });
  std::for_each(cloud2->points.begin(), cloud2->points.end(),
                [&pc2](const pcl_PointType &pt) {
                  pc2.emplace_back(pt.getVector3fMap().cast<double>());
                });
  KDTree<double, 3> kdtree;
  auto func_build = [&kdtree, &pc1]() { kdtree.setInputCloud(pc1); };
  call_evaluate(func_build, "func_build", 5);
  EXPECT_EQ(kdtree.leaf_size(), cloud1->size());

  std::vector<std::vector<int>> pcl_match_idx;
  std::vector<std::vector<float>> pcl_match_dist;

  const size_t k = 5;
  std::vector<int> query_idx(cloud2->size());
  std::iota(query_idx.begin(), query_idx.end(), 0);
  auto func_pcl = [&]() {
    kdtree_pcl.nearestKSearch(*cloud2, query_idx, k, pcl_match_idx,
                              pcl_match_dist);
  };
  call_evaluate(func_pcl, "func_pcl", 5);

  std::vector<std::vector<int>> knn_match_idx;
  std::vector<std::vector<double>> knn_match_dist;
  auto func_knn_mt = [&]() {
    kdtree.nearest_neighbors_kmt(pc2, k, knn_match_idx, knn_match_dist);
  };
  call_evaluate(func_knn_mt, "func_knn_mt", 5);

  for (size_t qid = 0; qid < query_idx.size(); ++qid) {
    EXPECT_TRUE(std::equal(pcl_match_idx[qid].begin(), pcl_match_idx[qid].end(),
                           knn_match_idx[qid].begin()));
    for (size_t rid = 0; rid < k; ++rid) {
      EXPECT_LE(std::abs(pcl_match_dist[qid][rid] - knn_match_dist[qid][rid]),
                0.01);
    }
  }

  kdtree_pcl.setInputCloud(cloud2);
  kdtree.setInputCloud(pc2);
  pcl_match_idx.clear();
  pcl_match_dist.clear();
  query_idx.resize(cloud1->size());
  std::iota(query_idx.begin(), query_idx.end(), 0);

  auto func_pcl2 = [&]() {
    kdtree_pcl.nearestKSearch(*cloud1, query_idx, k, pcl_match_idx,
                              pcl_match_dist);
  };
  call_evaluate(func_pcl2, "func_pcl2", 5);

  knn_match_idx.clear();
  knn_match_dist.clear();
  auto func_knn_mt2 = [&]() {
    kdtree.nearest_neighbors_kmt(pc1, k, knn_match_idx, knn_match_dist);
  };
  call_evaluate(func_knn_mt2, "func_knn_mt2", 5);

  for (size_t qid = 0; qid < query_idx.size(); ++qid) {
    EXPECT_TRUE(std::equal(pcl_match_idx[qid].begin(), pcl_match_idx[qid].end(),
                           knn_match_idx[qid].begin()));
    for (size_t rid = 0; rid < k; ++rid) {
      EXPECT_LE(std::abs(pcl_match_dist[qid][rid] - knn_match_dist[qid][rid]),
                0.01);
    }
  }
}

TEST(KDTest, kNN2D) {
  using PointCloud = KDTree<double, 2>::PointCloud;
  using pcl_PointType = pcl::PointXY;
  using pcl_PointCloud = pcl::PointCloud<pcl_PointType>;

  pcl::PointCloud<pcl::PointXYZI>::Ptr c1(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr c2(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(FLAGS_pcd_path1, *c1);
  pcl::io::loadPCDFile(FLAGS_pcd_path2, *c2);

  pcl_PointCloud::Ptr cloud1(new pcl_PointCloud);
  pcl_PointCloud::Ptr cloud2(new pcl_PointCloud);
  PointCloud pc1, pc2;
  pc1.reserve(cloud1->size());
  pc2.reserve(cloud2->size());

  std::for_each(c1->points.begin(), c1->points.end(),
                [&cloud1, &pc1](const pcl::PointXYZI &pt) {
                  cloud1->points.emplace_back(pcl::PointXY(pt.x, pt.y));
                  pc1.emplace_back(pt.x, pt.y);
                });
  std::for_each(c2->points.begin(), c2->points.end(),
                [&cloud2, &pc2](const pcl::PointXYZI &pt) {
                  cloud2->points.emplace_back(pcl::PointXY(pt.x, pt.y));
                  pc2.emplace_back(pt.x, pt.y);
                });

  LOG(INFO) << "Number of point in cloud1: " << cloud1->size();
  LOG(INFO) << "Number of point in cloud2: " << cloud2->size();

  pcl::search::KdTree<pcl_PointType> kdtree_pcl;
  auto func_pcl_build = [&kdtree_pcl, &cloud1]() {
    kdtree_pcl.setInputCloud(cloud1);
  };
  call_evaluate(func_pcl_build, "func_pcl_build", 5);

  KDTree<double, 2> kdtree;
  auto func_build = [&kdtree, &pc1]() { kdtree.setInputCloud(pc1); };
  call_evaluate(func_build, "func_build", 5);
  EXPECT_EQ(kdtree.leaf_size(), cloud1->size());

  std::vector<std::vector<int>> pcl_match_idx;
  std::vector<std::vector<float>> pcl_match_dist;

  const size_t k = 5;
  std::vector<int> query_idx(cloud2->size());
  std::iota(query_idx.begin(), query_idx.end(), 0);
  auto func_pcl = [&]() {
    kdtree_pcl.nearestKSearch(*cloud2, query_idx, k, pcl_match_idx,
                              pcl_match_dist);
  };
  call_evaluate(func_pcl, "func_pcl", 5);

  std::vector<std::vector<int>> knn_match_idx;
  std::vector<std::vector<double>> knn_match_dist;
  auto func_knn_mt = [&]() {
    kdtree.nearest_neighbors_kmt(pc2, k, knn_match_idx, knn_match_dist);
  };
  call_evaluate(func_knn_mt, "func_knn_mt", 5);

  for (size_t qid = 0; qid < query_idx.size(); ++qid) {
    EXPECT_TRUE(std::equal(pcl_match_idx[qid].begin(), pcl_match_idx[qid].end(),
                           knn_match_idx[qid].begin()));
    for (size_t rid = 0; rid < k; ++rid) {
      EXPECT_LE(std::abs(pcl_match_dist[qid][rid] - knn_match_dist[qid][rid]),
                0.01);
    }
  }

  kdtree_pcl.setInputCloud(cloud2);
  kdtree.setInputCloud(pc2);
  pcl_match_idx.clear();
  pcl_match_dist.clear();
  query_idx.resize(cloud1->size());
  std::iota(query_idx.begin(), query_idx.end(), 0);

  auto func_pcl2 = [&]() {
    kdtree_pcl.nearestKSearch(*cloud1, query_idx, k, pcl_match_idx,
                              pcl_match_dist);
  };
  call_evaluate(func_pcl2, "func_pcl2", 5);

  knn_match_idx.clear();
  knn_match_dist.clear();
  auto func_knn_mt2 = [&]() {
    kdtree.nearest_neighbors_kmt(pc1, k, knn_match_idx, knn_match_dist);
  };
  call_evaluate(func_knn_mt2, "func_knn_mt2", 5);

  for (size_t qid = 0; qid < query_idx.size(); ++qid) {
    EXPECT_TRUE(std::equal(pcl_match_idx[qid].begin(), pcl_match_idx[qid].end(),
                           knn_match_idx[qid].begin()));
    for (size_t rid = 0; rid < k; ++rid) {
      EXPECT_LE(std::abs(pcl_match_dist[qid][rid] - knn_match_dist[qid][rid]),
                0.01);
    }
  }
}
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}