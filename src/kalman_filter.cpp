#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

static MatrixXd Id4 = MatrixXd::Identity(4, 4);

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = P_ * Ht * Sinv;
  x_ += K * y;
  P_ = (Id4 - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateCommon(y);
}

VectorXd h(const VectorXd & x) {
  double px = x(0);
  double py = x(1);
  double vx = x(2);
  double vy = x(3);

  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / rho;

  VectorXd result(3);
  result << rho, phi, rho_dot;
  return result;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = z - h(x_);
  while (y(1) > M_PI) y(1) -= 2 * M_PI;
  while (y(1) < -M_PI) y(1) += 2 * M_PI;
  UpdateCommon(y);
}
