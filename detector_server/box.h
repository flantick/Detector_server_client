#ifndef Box_H
#define Box_H

#pragma once

class Box {
public:
    unsigned short x1, y1, x2, y2;
    float conf;
    int label;
    Box(
        unsigned short x1,
        unsigned short y1,
        unsigned short x2,
        unsigned short y2,
        float conf,
        int label
    );
};
#endif