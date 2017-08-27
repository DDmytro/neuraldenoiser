#ifndef IMAGES_H
#define IMAGES_H

#include <iostream>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

inline void swap(float& a, float& b)
{
    auto temp = a;
    a = b;
    b = temp;
}

struct RGBImage
{
    float* pixels;
    int nX;
    int nY;

    inline int getW(){return nX;}
    inline int getH(){return nY;}

    RGBImage(const char* file);

    RGBImage(int w, int h)
    {
        nX = w;
        nY = h;
        pixels = new float[w * h * 3];
        memset(pixels, 0, w * h * 3 * sizeof(float));
    }

    void fill(float r, float g, float b)
    {
        for (int i = 0; i < nX * nY; i++)
        {
            pixels[i * 3] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }

    }

    ~RGBImage()
    {
        delete[] pixels;
    }

    void getRect(float* pix, int x, int y, int w, int h)
    {
        for (int j = 0; j < h; j++)
        {
            int Y = y + j;
            if (Y < 0) Y = - Y - 1;
            if (Y >= nY) Y =  2 * nY - Y - 1;
            for (int i = 0; i < w; i++)
            {
                int X = x + i;
                if (X < 0) X = - X - 1;
                if (X >= nX) X =  2 * nX - X - 1;

                pix[i * 3 + j * w * 3] = pixels[X * 3 + Y * nX * 3];
                pix[i * 3 + 1 + j * w * 3] = pixels[X * 3 + 1 + Y * nX * 3];
                pix[i * 3 + 2 + j * w * 3] = pixels[X * 3 + 2 + Y * nX * 3];
            }
        }
    }

    void setRect(float* pix, int x, int y, int w, int h)
    {
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w * 3; i++)
                 pixels[x*3 + i + (y + j) * nX * 3] = pix[i + j * w*3];
    }

    void setPixel(float* rgb, int x, int y)
    {
         int idx = (x + y * nX) * 3;
         pixels[idx++] = *rgb++;
         pixels[idx++] = *rgb++;
         pixels[idx++] = *rgb++;
    }

    void normalize()
    {
        float max = pixels[0];
        float min = pixels[0];
        for (int i = 1; i < nX * nY * 3; i++)
        {
            if (pixels[i] > max) max = pixels[i];
            if (pixels[i] < min) min = pixels[i];
        }

        for (int i = 0; i < nX * nY * 3; i++)
             pixels[i] = (pixels[i] - min) / (max - min);
    }

    void applyGamma(float g)
    {
        g = 1 / g;
        for (int i = 0; i < nX * nY * 3; i++)
             pixels[i] = pow(pixels[i], g);
    }

    void flipHor()
    {
        for (int j = 0; j < nY; j++)
            for (int i = 0; i < nX / 2; i++)
            {
                swap(pixels[(i + j * nX) * 3], pixels[(nX - i - 1 + j * nX) * 3]);
                swap(pixels[(i + j * nX) * 3 + 1], pixels[(nX - i - 1 + j * nX) * 3 + 1]);
                swap(pixels[(i + j * nX) * 3 + 2], pixels[(nX - i - 1 + j * nX) * 3 + 2]);
            }
    }

    void flipVert()
    {
        for (int j = 0; j < nY / 2; j++)
            for (int i = 0; i < nX; i++)
            {
                swap(pixels[(i + j * nX) * 3], pixels[(i + (nY - j - 1) * nX) * 3]);
                swap(pixels[(i + j * nX) * 3 + 1], pixels[(i + (nY - j - 1) * nX) * 3 + 1]);
                swap(pixels[(i + j * nX) * 3 + 2], pixels[(i + (nY - j - 1) * nX) * 3 + 2]);
            }
    }

    void Save(const char* name);
};

struct GRAYImage
{
    float* pixels;
    int nX;
    int nY;
    GRAYImage(const char* file);
    GRAYImage(int w, int h)
    {
        nX = w;
        nY = h;
        pixels = new float[w * h];
    }

    void getRect(float* pix, int x, int y, int w, int h)
    {
        for (int j = 0; j < h; j++)
        {
            int Y = y + j;
            if (Y < 0) Y = - Y - 1;
            if (Y >= nY) Y =  2 * nY - Y - 1;
            for (int i = 0; i < w; i++)
            {
                int X = x + i;
                if (X < 0) X = - X - 1;
                if (X >= nX) X =  2 * nX - X - 1;
                pix[i + j * w] = pixels[X + Y * nX];
            }
        }
    }

    void normalize()
    {
        float max = pixels[0];
        for (int i = 1; i < nX * nY; i++)
            if (pixels[i] > max) max = pixels[i];

        for (int i = 0; i < nX * nY; i++)
             pixels[i] /= max * 1.2;

    }

    void flipHor()
    {
        for (int j = 0; j < nY; j++)
            for (int i = 0; i < nX / 2; i++)
                swap(pixels[i + j * nX], pixels[nX - i - 1 + j * nX]);
    }

    void flipVert()
    {
        for (int j = 0; j < nY / 2; j++)
            for (int i = 0; i < nX; i++)
                swap(pixels[i + j * nX], pixels[i + (nY - j - 1) * nX]);
    }

    ~GRAYImage()
    {
        delete[] pixels;
    }
};

GRAYImage::GRAYImage(const char *file)
{
    float* rgba;
    if (TINYEXR_SUCCESS != LoadEXR(&rgba, &nX, &nY, file, NULL))
    {
        std::cout << "Error loading" << std::endl;
    }
    pixels = new float[nX * nY];
    for (int j = 0; j < nY; j++)
        for (int i = 0; i < nX; i++)
        {
            pixels[i + j * nX] =  rgba[(i + j * nX) * 4];
        }
    delete[] rgba;
}

RGBImage::RGBImage(const char *file)
{
    float* rgba;
    if (TINYEXR_SUCCESS != LoadEXR(&rgba, &nX, &nY, file, NULL))
    {
        std::cout << "Error loading" << std::endl;
    }
    pixels = new float[nX * nY * 3];
    for (int j = 0; j < nY; j++)
        for (int i = 0; i < nX; i++)
        {
            pixels[(i + j * nX) * 3] =  rgba[(i + j * nX) * 4];
            pixels[(i + j * nX) * 3 + 1] =  rgba[(i + j * nX) * 4 + 1];
            pixels[(i + j * nX) * 3 + 2] =  rgba[(i + j * nX) * 4 + 2];
        }
    delete[] rgba;
}

void RGBImage::Save(const char* name)
{
    SaveEXR((float*)pixels, nX, nY, 3, (std::string(name) + ".exr").c_str());
}
#endif // IMAGES_H
