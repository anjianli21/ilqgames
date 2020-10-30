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
// Three player overtaking example.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ILQGAMES_EXAMPLE_HIGHWAY_MERGING_EXAMPLE_H
#define ILQGAMES_EXAMPLE_HIGHWAY_MERGING_EXAMPLE_H

#include <ilqgames/dynamics/multi_player_flat_system.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/solver/top_down_renderable_problem.h>

namespace ilqgames {

class HighwayMergingExample : public TopDownRenderableProblem {
public:
  ~HighwayMergingExample() {}

  HighwayMergingExample(Time adversarial_time = 0.0)
      : TopDownRenderableProblem(adversarial_time) {}

  // Construct dynamics, initial state, and player costs.
  void ConstructDynamics();
  void ConstructInitialState();
  void ConstructPlayerCosts();
  // void ConstructInitialOperatingPoint();

  //  void SetAdversarialTime(double adv_time);

  // Unpack x, y, heading (for each player, potentially) from a given state.
  std::vector<float> Xs(const VectorXf &x) const;
  std::vector<float> Ys(const VectorXf &x) const;
  std::vector<float> Thetas(const VectorXf &x) const;

  // private:
  //  double adversarial_time;
}; // class ThreePlayerIntersectionExample

} // namespace ilqgames

#endif