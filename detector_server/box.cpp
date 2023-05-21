#include "box.h"


Box::Box(
    unsigned short x1,
    unsigned short y1,
    unsigned short x2,
    unsigned short y2,
    float conf,
    int label
) {
    this->x1 = x1;
    this->y1 = y1;
    this->x2 = x2;
    this->y2 = y2;
    this->conf = conf;
    this->label = label;
};