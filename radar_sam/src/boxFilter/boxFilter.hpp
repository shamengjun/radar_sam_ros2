#include "utility.h"

class BoxFilter
{
public:
    // using PointT = pcl::PointXYZI;
    BoxFilter(std::vector<double> size);
    BoxFilter() = default;
    bool filter(const pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr &input, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr &output);
    void setSize(std::vector<double> size);
    void setOrigin(std::vector<double> origin);
    std::vector<double> getEdge();

private:
    void calculateEdge();

private:
    pcl::CropBox<XYZRGBSemanticsInfo> pcl_box_filter_;
    std::vector<double> origin_;
    std::vector<double> size_;
    std::vector<double> edge_;
};

BoxFilter::BoxFilter(std::vector<double> size)
{
    size_.resize(6);
    edge_.resize(6);
    origin_.resize(3);
    for (size_t i = 0; i < 6; ++i)
    {
        size_.at(i) = size[i];
    }
    setSize(size_);
}

bool BoxFilter::filter(const pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr &input, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr &output)
{
    if (output->size() != 0)
        output->clear();
    std::cout << "input cloud size:" << input->size() << std::endl;
    pcl_box_filter_.setMin(Eigen::Vector4f(edge_.at(0), edge_.at(2), edge_.at(4), 1.0e-6));
    pcl_box_filter_.setMax(Eigen::Vector4f(edge_.at(1), edge_.at(3), edge_.at(5), 1.0e6));
    pcl_box_filter_.setInputCloud(input);
    pcl_box_filter_.filter(*output);
    std::cout << "after filter cloud size:" << output->size() << std::endl;
    if (output->size() == 0)
        return false;
    return true;
}

void BoxFilter::setSize(std::vector<double> size)
{
    size_ = size;
    std::cout << "Box filter params:" << std::endl
              << "min_x:" << size_.at(0) << ", "
              << "max_x:" << size_.at(1) << ","
              << "min_y:" << size_.at(2) << ", "
              << "max_y:" << size_.at(3) << ","
              << "min_z:" << size_.at(4) << ", "
              << "max_z:" << size_.at(5) << std::endl;
    calculateEdge();
}

void BoxFilter::setOrigin(std::vector<double> origin)
{
    origin_ = origin;
    calculateEdge();
}

void BoxFilter::calculateEdge()
{
    for (size_t i = 0; i < origin_.size(); ++i)
    {
        edge_.at(2 * i) = size_.at(2 * i) + origin_.at(i);
        edge_.at(2 * i + 1) = size_.at(2 * i + 1) + origin_.at(i);
    }
}

std::vector<double> BoxFilter::getEdge()
{
    return edge_;
}