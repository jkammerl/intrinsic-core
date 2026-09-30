// Copyright 2026 Intrinsic Innovation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "intrinsic/perception/core/compute_normals_least_sqr.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "intrinsic/perception/core/image.h"
#include "intrinsic/perception/core/image_traits.h"
#include "intrinsic/perception/core/intrinsic_params.h"

namespace intrinsic::perception {
namespace {

constexpr int32_t kCols = 64, kRows = 48;
constexpr double kFocal = 50.0;

// Renders the depth image of the plane {p : dot(normal, p) == offset}.
Image<Depth32f> PlaneDepth(const IntrinsicParams& params, const double n[3],
                           double offset) {
  Image<Depth32f> depth(kCols, kRows);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kCols; ++col) {
      const double ray[3] = {
          (col - params.principal_point_x()) / params.focal_length_x(),
          (row - params.principal_point_y()) / params.focal_length_y(), 1.0};
      depth(col, row) =
          offset / (n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2]);
    }
  }
  return depth;
}

// Checks the normals in the processed region against the plane's normal.
void ExpectPlaneNormals(const double n[3]) {
  const IntrinsicParams params(Dimensions(kCols, kRows), kFocal, kFocal);
  const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  const int32_t radius = 2;
  Image<Normal32f> normals = ComputeNormalsLeastSqr(
      params, PlaneDepth(params, n, -1.0), /*threshold=*/0.5, radius,
      /*step=*/1);
  int32_t valid = 0;
  for (int32_t row = radius; row < kRows - radius; ++row) {
    for (int32_t col = radius; col < kCols - radius - 8; ++col) {
      const auto& normal = normals(col, row);
      ASSERT_TRUE(Normal32f::IsValid(normal)) << col << ", " << row;
      // Unit length, facing the camera (negative z), parallel to the plane's
      // normal.
      const double dot =
          (normal.x() * n[0] + normal.y() * n[1] + normal.z() * n[2]) / norm;
      EXPECT_NEAR(normal.x() * normal.x() + normal.y() * normal.y() +
                      normal.z() * normal.z(),
                  1.0, 1e-2);
      EXPECT_LT(normal.z(), 0);
      EXPECT_GT(std::abs(dot), 0.999) << col << ", " << row;
      ++valid;
    }
  }
  EXPECT_GT(valid, 0);
  // Pixels outside the processed region stay invalid.
  EXPECT_FALSE(Normal32f::IsValid(normals(0, 0)));
  EXPECT_FALSE(Normal32f::IsValid(normals(kCols - 1, kRows / 2)));
}

TEST(ComputeNormalsLeastSqrTest, FrontoParallelPlane) {
  const double n[3] = {0.0, 0.0, -1.0};
  ExpectPlaneNormals(n);
}

TEST(ComputeNormalsLeastSqrTest, TiltedPlane) {
  const double n[3] = {0.3, -0.2, -1.0};
  ExpectPlaneNormals(n);
}

TEST(ComputeNormalsLeastSqrTest, NoValidNeighborsGivesInvalidNormal) {
  const IntrinsicParams params(Dimensions(kCols, kRows), kFocal, kFocal);
  Image<Depth32f> depth(Dimensions(kCols, kRows),
                        std::numeric_limits<float>::quiet_NaN());
  Image<Normal32f> normals = ComputeNormalsLeastSqr(params, depth, 0.5, 2, 1);
  EXPECT_FALSE(Normal32f::IsValid(normals(kCols / 2, kRows / 2)));
}

}  // namespace
}  // namespace intrinsic::perception
