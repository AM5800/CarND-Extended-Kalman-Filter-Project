#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

struct KalmanFilter {
  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  /**
  * @param state_len Expected length of the state vector x
  */
  KalmanFilter(int state_len);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict(const Eigen::MatrixXd &F, const Eigen::MatrixXd& Q);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H, const Eigen::MatrixXd& R);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z, const Eigen::MatrixXd &H, const Eigen::MatrixXd& R);

private:
  void UpdateCommon(const Eigen::VectorXd &y, const Eigen::MatrixXd &H, const Eigen::MatrixXd& R);
  Eigen::MatrixXd identity_;
};

#endif /* KALMAN_FILTER_H_ */
