#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter(int state_len) : x_(VectorXd::Zero(state_len)),
                                            P_(1000 * MatrixXd::Identity(state_len, state_len)),
                                            identity_(MatrixXd::Identity(state_len, state_len)) {

}


void KalmanFilter::Predict(const MatrixXd& F, const MatrixXd& Q) {
  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::UpdateCommon(const VectorXd& y, const MatrixXd& H, const MatrixXd& R) {
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Sinv = S.inverse();
  MatrixXd K = P_ * Ht * Sinv;
  x_ += K * y;
  P_ = (identity_ - K * H) * P_;
}

void KalmanFilter::Update(const VectorXd& z, const MatrixXd& H, const MatrixXd& R) {
  VectorXd y = z - H * x_;
  UpdateCommon(y, H, R);
}

VectorXd h(const VectorXd& x) {
  double px = x(0);
  double py = x(1);
  double vx = x(2);
  double vy = x(3);

  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / rho;

  VectorXd result(3);
  result << rho , phi , rho_dot;
  return result;
}

void KalmanFilter::UpdateEKF(const VectorXd& z, const MatrixXd& H, const MatrixXd& R) {
  VectorXd y = z - h(x_);
  while (y(1) > M_PI) y(1) -= 2 * M_PI;
  while (y(1) < -M_PI) y(1) += 2 * M_PI;
  UpdateCommon(y, H, R);
}
