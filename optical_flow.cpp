#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;


string file_1 = "../1.png";  // 第一张图片
string file_2 = "../2.png";  // 第二张图片

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}


int main(int argc, char **argv) {

    // 图片，格式为CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // GFTT提取角点
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // 用单层光流进行追踪
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, false);

    // 用多层光流进行追踪
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, false);

    // 用opecv光流进行追踪
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // 绘制
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++)
    {
        if (success_single[i])
        {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {

    // 参数
    int half_patch_size = 4;
    int iterations = 10;  //一般迭代不会超过10次就越界了
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy 需要估计
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        // Gauss-Newton迭代
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }
            // 计算cost和jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error = 0;
                    error=GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y)-GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy);

                    Eigen::Vector2d J;  // Jacobian
                    if (inverse == false) {
                        // 前向光流
                        if(kp.pt.x + x + 1 + dx>=0 && kp.pt.x + x -1 + dx<=img1.cols && kp.pt.y + y + 1 + dy>=0 && kp.pt.y + y + 1 + dy<=img1.rows
                           &&kp.pt.x + x - 1 + dx>=0 && kp.pt.x + x - 1 + dx<=img1.cols && kp.pt.y + y - 1 + dy>=0 && kp.pt.y + y - 1 + dy<=img1.rows) {
                            double i2_x_plus1 = GetPixelValue(img2, kp.pt.x + x + 1 + dx, kp.pt.y + y + dy);
                            double i2_x_minus1 = GetPixelValue(img2, kp.pt.x + x - 1 + dx, kp.pt.y + y + dy);
                            double i2_y_plus1 = GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + 1 + dy);
                            double i2_y_minus1 = GetPixelValue(img2, kp.pt.x + x + dx,
                                                               kp.pt.y + y - 1 + dy); //根据导数定义推导要加dx,dy.

                            double derivative_i2_x = -(i2_x_plus1 - i2_x_minus1) / 2;
                            double derivative_i2_y = -(i2_y_plus1 - i2_y_minus1) / 2;
                            J(0) = derivative_i2_x;
                            J(1) = derivative_i2_y;
                        }
                    } else
                    {
                        // 反向光流
                        if(kp.pt.x + x + 1>=0 && kp.pt.x + x + 1<=img1.cols && kp.pt.y + y + 1>=0 && kp.pt.y + y + 1<=img1.rows
                           &&kp.pt.x + x - 1>=0 && kp.pt.x + x - 1<=img1.cols && kp.pt.y + y - 1>=0 && kp.pt.y + y - 1<=img1.rows)
                        {
                            double i1_x_plus1=GetPixelValue(img1,kp.pt.x + x + 1,kp.pt.y + y);
                            double i1_x_minus1=GetPixelValue(img1,kp.pt.x + x - 1,kp.pt.y + y);
                            double i1_y_plus1=GetPixelValue(img1,kp.pt.x + x,kp.pt.y + y+ 1);
                            double i1_y_minus1=GetPixelValue(img1,kp.pt.x + x,kp.pt.y + y- 1);

                            double derivative_i1_x=-(i1_x_plus1-i1_x_minus1)/2;
                            double derivative_i1_y=-(i1_y_plus1-i1_y_minus1)/2;
                            J(0)=derivative_i1_x;
                            J(1)=derivative_i1_y;
                        }
                    }

                    // 计算H, b
                    H += J*J.transpose();
                    b += -error*J;
                    cost +=error*error;  //是+=不是=，注意
                }

            // 迭代更新
            Eigen::Vector2d update;
            update=H.ldlt().solve(b);

            if (std::isnan(update[0]))  //std::isnan,如果直接用isnan会报错
            {
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // 更新 dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // 参数
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    // 创建金字塔
    vector<Mat> pyr1, pyr2; // 图像金字塔
    for (int i = 0; i < pyramids; i++) {
        Mat temp_1, temp_2;
        resize(img1, temp_1, Size(), scales[i], scales[i]);  //pyrDown()不能乘以过小的系数
        resize(img2, temp_2, Size(), scales[i], scales[i]);
        pyr1.push_back(temp_1);
        pyr2.push_back(temp_2);
    }

    // coarse-to-fine LK 光流金字塔
    for (int level = pyramids - 1; level >=0; level--) {
        vector<KeyPoint> kp1_pyr; // 设置金字塔层数
        KeyPoint temp;
        for (auto &kp1_i: kp1) {
            temp.pt=scales[level] * kp1_i.pt;
            kp1_pyr.push_back(temp);
        }
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2, success);
        if(level!=0)  //第一层不用乘以尺度
        {
            for(size_t i=0;i<kp2.size();i++)
            {
                kp2[i].pt=2*kp2[i].pt;
            }
        }
    }
}

