#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd>& estimations,
                              const vector<VectorXd>& ground_truth) {
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
    throw std::runtime_error("Estimation and Ground Truth size mismatch");

  VectorXd sum = VectorXd::Zero(estimations.front().rows());

  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = ground_truth[i] - estimations[i];
    VectorXd residual_squared = residual.array() * residual.array();
    sum += residual_squared;
  }

  return (sum / estimations.size()).array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd result = MatrixXd::Zero(3, 4);

  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double px_py = px * px + py * py;
  if (fabs(px_py) < 0.00001) throw std::runtime_error("Division by zero!");

  double sqrt_px_py = sqrt(px_py);

  result(0, 0) = px / sqrt_px_py;
  result(0, 1) = py / sqrt_px_py;
  result(1, 0) = -py / px_py;
  result(1, 1) = px / px_py;
  result(2, 0) = py * (vx * py - vy * px) / (sqrt_px_py * px_py);
  result(2, 1) = px * (vy * px - vx * py) / (sqrt_px_py * px_py);
  result(2, 2) = px / sqrt_px_py;
  result(2, 3) = py / sqrt_px_py;

  return result;
}
