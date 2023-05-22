#include "utilities.h"

using namespace std;

std::string COCO[80] = {
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"dining table",
"toilet",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
};

cv::Mat coverImg(cv::Mat& img, cv::Size trgSize) {
    cv::Mat imgROI; img.copyTo(imgROI);
    if (imgROI.size[1] > trgSize.width)
        cv::resize(imgROI, imgROI,
            { trgSize.width, trgSize.height * imgROI.size[0] / imgROI.size[1] }
    );
    if (imgROI.size[0] > trgSize.height)
        cv::resize(imgROI, imgROI,
            { trgSize.width * imgROI.size[1] / imgROI.size[0], trgSize.height }
    );
    cv::Mat canvas = cv::Mat::zeros(trgSize, CV_8UC3);
    cv::Mat canvasROI = canvas(
        cv::Rect(
            (trgSize.width - imgROI.size[1]) / 2, (trgSize.height - imgROI.size[0]) / 2,
            imgROI.size[1], imgROI.size[0]
        )
    );
    imgROI.copyTo(canvasROI);
    return canvas;
}

float calculateIoU(const Box& box1, const Box& box2) {
    unsigned short interX1 = std::max(box1.x1, box2.x1);
    unsigned short interY1 = std::max(box1.y1, box2.y1);
    unsigned short interX2 = std::min(box1.x2, box2.x2);
    unsigned short interY2 = std::min(box1.y2, box2.y2);

    unsigned short interWidth = std::max(interX2 - interX1, 0);
    unsigned short interHeight = std::max(interY2 - interY1, 0);

    unsigned int intersectionArea = interWidth * interHeight;

    unsigned int box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    unsigned int box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    unsigned int unionArea = box1Area + box2Area - intersectionArea;

    float iou = static_cast<float>(intersectionArea) / unionArea;
    return iou;
}

vector<Box> nms(const vector<Box>& boxes, float iouThres) {
    vector<Box> selectedBoxes;

    for (const Box& box : boxes) {
        bool keep = true;

        for (const Box& selectedBox : selectedBoxes) {
            float iou = calculateIoU(box, selectedBox);

            if (iou > iouThres) {
                keep = false;
                break;
            }
        }

        if (keep) {
            selectedBoxes.push_back(box);
        }
    }

    return selectedBoxes;
}

std::vector<Box> getBoxes(at::Tensor& outputs, float confThres = 0.25, float iouThres = 0.45
) {
    vector<Box> candidates;
    candidates.reserve(outputs.sizes()[0] * outputs.sizes()[2]);  // Reserve memory for candidates

    auto accessor = outputs.accessor<float, 3>();  // Create an accessor for efficient tensor access

    for (unsigned short ibatch = 0; ibatch < outputs.sizes()[0]; ibatch++) {
        for (unsigned short ibox = 0; ibox < outputs.sizes()[2]; ibox++) {
            float maxConf = 0.0;
            int class_ = -1;

            for (unsigned short iclass = 4; iclass < outputs.sizes()[1]; iclass++) {
                float conf = accessor[ibatch][iclass][ibox];
                if (conf > maxConf) {
                    maxConf = conf;
                    class_ = iclass - 4;
                }
            }

            if (maxConf >= confThres) {
                unsigned short
                    cx = accessor[ibatch][0][ibox],
                    cy = accessor[ibatch][1][ibox],
                    w = accessor[ibatch][2][ibox],
                    h = accessor[ibatch][3][ibox];

                unsigned short
                    x1 = cx - w / 2,
                    y1 = cy - h / 2,
                    x2 = cx + w / 2,
                    y2 = cy + h / 2;

                candidates.emplace_back(x1, y1, x2, y2, maxConf, class_);  // Use emplace_back for construction

            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Box& b1, const Box& b2) {
        return b1.conf > b2.conf;
        });

    vector<Box> boxes = nms(candidates, iouThres);

    return boxes;
}

void highlightBoxes(cv::Mat& img, vector<Box>& boxes) {

    cv::Scalar rectColor(0, 192, 0);
    unsigned short fontScale = 1, confPrecis = 2;

    for (Box box : boxes) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(confPrecis) << box.conf;
        std::string text = ss.str();
        std::string class_ = COCO[box.label];
        text = text + " - " + class_;
        cv::rectangle(img, { box.x1,box.y1 }, { box.x2,box.y2 }, rectColor, 2);
        cv::rectangle(
            img,
            { box.x1, box.y1 - fontScale * 12 },
            { box.x1 + (unsigned short)text.length() * fontScale * 9, box.y1 },
            rectColor,
            -1
        );
        cv::putText(img, text, { box.x1,box.y1 }, cv::FONT_HERSHEY_PLAIN, fontScale, { 255,255,255 }, 2);
    }
}

cv::Mat detect(
    torch::jit::script::Module& model,
    cv::Mat img,
    torch::Device& device,
    int imgMaxWidth
) {
    if (img.size[1] > imgMaxWidth)
        cv::resize(img, img, { imgMaxWidth, imgMaxWidth * img.size[0] / img.size[1] });

    cv::Mat imgCov = coverImg(img, { 640,640 });

    cv::Mat imgNorm; imgCov.copyTo(imgNorm);
    cv::cvtColor(imgNorm, imgNorm, cv::COLOR_BGR2RGB);
    cv::normalize(imgNorm, imgNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    at::Tensor inputTensor = torch::from_blob(
        imgNorm.data,
        { 640, 640, 3 },
        torch::kFloat32
    ).permute({ 2, 0, 1 }).unsqueeze(0).to(device);
    at::Tensor outputs = model.forward({ inputTensor }).toTensor();
    outputs = outputs.to(torch::kCPU);
    vector<Box> boxes = getBoxes(outputs);


    highlightBoxes(imgCov, boxes);
    return imgCov;
}