#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <gtsam/base/serialization.h>

// ... Includes for your values and factors:
#include <gtsam/base/GenericValue.h> // GTSAM_VALUE_EXPORT
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/expressions.h>
// ...

// Define the Boost export macros:
#include <Eigen/Dense>
#include <boost/serialization/export.hpp> // BOOST_CLASS_EXPORT_GUID
#include <boost/serialization/serialization.hpp>
#include <gtsam/base/GenericValue.h> // GTSAM_VALUE_EXPORT

namespace boost
{
  namespace serialization
  {
    template <class Archive, typename Derived>
    void serialize(Archive &ar, Eigen::EigenBase<Derived> &g, const unsigned int version)
    {
      ar & boost::serialization::make_array(g.derived().data(), g.size());
    }
  } // namespace serialization
} // namespace boost

/* Create GUIDs for Noisemodels */
/* ************************************************************************* */
// clang-format off
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Constrained, "gtsam_noiseModel_Constrained");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Diagonal, "gtsam_noiseModel_Diagonal");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Gaussian, "gtsam_noiseModel_Gaussian");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Unit, "gtsam_noiseModel_Unit");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Isotropic,"gtsam_noiseModel_Isotropic");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::Robust, "gtsam_noiseModel_Robust");

BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Base, "gtsam_noiseModel_mEstimator_Base");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Null,"gtsam_noiseModel_mEstimator_Null");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Fair, "gtsam_noiseModel_mEstimator_Fair");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Huber,"gtsam_noiseModel_mEstimator_Huber");
BOOST_CLASS_EXPORT_GUID(gtsam::noiseModel::mEstimator::Tukey, "gtsam_noiseModel_mEstimator_Tukey");

BOOST_CLASS_EXPORT_GUID(gtsam::SharedNoiseModel, "gtsam_SharedNoiseModel");
BOOST_CLASS_EXPORT_GUID(gtsam::SharedDiagonal, "gtsam_SharedDiagonal");
// clang-format on

/* Create GUIDs for geometry */
/* ************************************************************************* */
// Export all classes derived from Value

GTSAM_VALUE_EXPORT(gtsam::Point3)
GTSAM_VALUE_EXPORT(gtsam::Pose3)
GTSAM_VALUE_EXPORT(gtsam::Rot3)


/* Create GUIDs for factors */
/* ************************************************************************* */
// clang-format off

BOOST_CLASS_EXPORT_GUID(gtsam::ExpressionFactor<gtsam::Point3>, "gtsam::ExpressionFactor<gtsam::Point3>");
BOOST_CLASS_EXPORT_GUID(gtsam::ExpressionFactor<gtsam::Rot3>, "gtsam::ExpressionFactor<gtsam::Rot3>");

// Add your custom factors, if any.

BOOST_CLASS_EXPORT_GUID(gtsam::PriorFactor<gtsam::Pose3>, "gtsam::PriorFactor<gtsam::Pose3>");
BOOST_CLASS_EXPORT_GUID(gtsam::BetweenFactor<gtsam::Pose3>, "gtsam::BetweenFactor<gtsam::Pose3>");
BOOST_CLASS_EXPORT_GUID(gtsam::GPSFactor, "gtsam::GPSFactor");

BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsam::JacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::HessianFactor , "gtsam::HessianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::GaussianConditional , "gtsam::GaussianConditional");



void saveBinary(const std::string &outFileName, const gtsam::NonlinearFactorGraph &f, const gtsam::Values &v) 
{
  std::ofstream ofs(outFileName);
  boost::archive::binary_oarchive oa(ofs);
  oa << f << v;
}

void loadBinary(const std::string &inFileName, gtsam::NonlinearFactorGraph &f, gtsam::Values &v) 
{
  std::ifstream ifs(inFileName);
  if (!ifs.is_open())
    throw std::runtime_error("Error opening file");

  boost::archive::binary_iarchive ia(ifs);
  ia >> f >> v;
}




