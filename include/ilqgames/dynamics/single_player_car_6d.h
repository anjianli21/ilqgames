/*
 * Copyright (c) 2019, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Single player dynamics modeling a car. 6 states and 2 control inputs.
// State is [x, y, theta, phi, v, a], control is [omega, j], and dynamics are:
//                     \dot px    = v cos theta
//                     \dot py    = v sin theta
//                     \dot theta = (v / L) * tan phi
//                     \dot phi   = omega
//                     \dot v     = a
//                     \dot a     = j
// Please refer to
// https://www.sciencedirect.com/science/article/pii/S2405896316301215
// for further details.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_DYNAMICS_SINGLE_PLAYER_CAR_6D_H
#define ILQGAMES_DYNAMICS_SINGLE_PLAYER_CAR_6D_H

#include <ilqgames/dynamics/single_player_dynamical_system.h>
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SinglePlayerCar6D : public SinglePlayerDynamicalSystem {
 public:
  ~SinglePlayerCar6D() {}
  SinglePlayerCar6D(float inter_axle_distance)
      : SinglePlayerDynamicalSystem(kNumXDims, kNumUDims),
        inter_axle_distance_(inter_axle_distance) {}

  // Compute time derivative of state.
  VectorXf Evaluate(Time t, const VectorXf& x, const VectorXf& u) const;

  // Compute a discrete-time Jacobian linearization.
  void Linearize(Time t, Time time_step, const VectorXf& x, const VectorXf& u,
                 Eigen::Ref<MatrixXf> A, Eigen::Ref<MatrixXf> B) const;

  // Distance metric between two states.
  float DistanceBetween(const VectorXf& x0, const VectorXf& x1) const;

  // Constexprs for state indices.
  static const Dimension kNumXDims;
  static const Dimension kPxIdx;
  static const Dimension kPyIdx;
  static const Dimension kThetaIdx;
  static const Dimension kPhiIdx;
  static const Dimension kVIdx;
  static const Dimension kAIdx;

  // Constexprs for control indices.
  static const Dimension kNumUDims;
  static const Dimension kOmegaIdx;
  static const Dimension kJerkIdx;

 private:
  // Inter-axle distance. Determines turning radius.
  const float inter_axle_distance_;
};  //\class SinglePlayerCar6D

// ----------------------------- IMPLEMENTATION ----------------------------- //

inline VectorXf SinglePlayerCar6D::Evaluate(Time t, const VectorXf& x,
                                            const VectorXf& u) const {
  VectorXf xdot(xdim_);
  xdot(kPxIdx) = x(kVIdx) * std::cos(x(kThetaIdx));
  xdot(kPyIdx) = x(kVIdx) * std::sin(x(kThetaIdx));
  xdot(kThetaIdx) = (x(kVIdx) / inter_axle_distance_) * std::tan(x(kPhiIdx));
  xdot(kPhiIdx) = u(kOmegaIdx);
  xdot(kVIdx) = x(kAIdx);
  xdot(kAIdx) = u(kJerkIdx);

  return xdot;
}

inline void SinglePlayerCar6D::Linearize(Time t, Time time_step,
                                         const VectorXf& x, const VectorXf& u,
                                         Eigen::Ref<MatrixXf> A,
                                         Eigen::Ref<MatrixXf> B) const {
  const float ctheta = std::cos(x(kThetaIdx)) * time_step;
  const float stheta = std::sin(x(kThetaIdx)) * time_step;
  const float cphi = std::cos(x(kPhiIdx));
  const float tphi = std::tan(x(kPhiIdx));

  A(kPxIdx, kThetaIdx) += -x(kVIdx) * stheta;
  A(kPxIdx, kVIdx) += ctheta;

  A(kPyIdx, kThetaIdx) += x(kVIdx) * ctheta;
  A(kPyIdx, kVIdx) += stheta;

  A(kThetaIdx, kPhiIdx) +=
      x(kVIdx) * time_step / (inter_axle_distance_ * cphi * cphi);
  A(kThetaIdx, kVIdx) += tphi * time_step / inter_axle_distance_;

  A(kVIdx, kAIdx) += time_step;

  B(kPhiIdx, kOmegaIdx) = time_step;
  B(kAIdx, kJerkIdx) = time_step;
}

inline float SinglePlayerCar6D::DistanceBetween(const VectorXf& x0,
                                                const VectorXf& x1) const {
  // Squared distance in position space.
  const float dx = x0(kPxIdx) - x1(kPxIdx);
  const float dy = x0(kPyIdx) - x1(kPyIdx);
  return dx * dx + dy * dy;
}

}  // namespace ilqgames

#endif
