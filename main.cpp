#include <stdlib.h>
#include <math.h>
#include <images.h>
#include <fstream>
#include <limits>
#include <time.h>

#define LOAD 1
#define sp 0.00001
#define gamma 2.4
#define PERIOD 16000

//ReLU and derivative computation
#define ACTIVATION(s) fmax(0.1f * s, s)
#define DERIVATIVE(s) (s < 0) ? 0.1f :  1.0f

//ADAM props
#define b1 0.9f
#define b2 0.999f

const std::string SAMPLES_FOLDER = "input";

float generateGaussianNoise(double mu, double sigma)
{
    static const double epsilon = std::numeric_limits<double>::min();
    static const double two_pi = 2.0*3.14159265358979323846;

    thread_local double z1;
    thread_local bool generate;
    generate = !generate;

    if (!generate)
       return z1 * sigma + mu;

    double u1, u2;
    do
     {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
     }
    while ( u1 <= epsilon );

    double z0;
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}


inline float fastabs(float f)
{
    int i=((*(int*)&f)&0x7fffffff);
    return (*(float*)&i);
}

// Fast inverse squire root
inline float InvSqrt(float x)
{
    float xhalf = 0.5f * x;
    int i = *(int*)&x;              // get bits for floating value
    i = 0x5f375a86 - (i >> 1);      // gives initial guess y0
    x = *(float*)&i;                // convert bits back to float
    x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
    x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
    return x;
}

#define ADAMAX_CORRECT(w, s, mt, vt, grad) \
{ \
   mt = b1 * mt + (1.0f - b1) * grad; \
   vt = fmaxf(b2 * vt, fastabs(grad));  \
   w += s * mt / (vt * (1 - b1t)); \
}

#define ADAM_CORRECT(w, s, mt, vt, grad) \
{ \
   mt = b1 * mt + (1.0f - b1) * grad; \
   vt = b2 * vt + (1.0f - b2) * grad * grad; \
   w += at * mt * InvSqrt(vt + 1e-8f); \
}

#define MOMENTUM_CORRECT(w, s, mt, vt, grad) \
{ \
    mt = b1 * mt + (1 - b1) * grad; \
    w += s * mt; \
}

#define RMSprop_CORRECT(w, s, mt, vt, grad) \
{ \
    vt = 0.9 * vt + 0.1 * grad * grad; \
    w += s * grad / sqrt(vt+1e-8); \
}


#define SIMPLE_CORRECT(w, s, grad) \
{ \
    w += s * grad; \
} \

template <int inpNum, int neironsNum>
class Layer
{
    float s;
    float* w = nullptr;
    float* der = nullptr;

    float* mt = nullptr;
    float* vt = nullptr;

    float b1t;
    float b2t;

    int step = 0;
    float getSpeed()
    {
        step++;
       // if (step < 16000 * 8)
            return s;
       // else return s * 4 / sqrt(step / 8000.f);
    }

public:
    Layer(float speed): s(speed)
    {
        b1t = 1;
        b2t = 1;
        w = new float[(inpNum + 1) * neironsNum];
        mt = new float[(inpNum + 1) * neironsNum];
        vt = new float[(inpNum + 1) * neironsNum];

        memset(mt, 0, sizeof(float) * (inpNum + 1) * neironsNum);
        memset(vt, 0, sizeof(float) * (inpNum + 1) * neironsNum);

        der = new float[neironsNum];
        for (int i = 0; i < (inpNum + 1) * neironsNum; i++)
            w[i] = generateGaussianNoise(0, 2.0 / inpNum);
    }

    ~Layer()
    {
        delete[] w;
        delete[] der;
    }

    void calc(float* in, float* out)
    {
        #pragma omp parallel  for
        for (int j = 0; j < neironsNum; j++)
        {
            float* wptr = w + j * (inpNum + 1);
            float sum = *wptr; // bias
            for (int i = 0; i < inpNum; i++)
                sum += wptr[i + 1] * in[i];
            out[j] = ACTIVATION(sum);
            der[j] = DERIVATIVE(sum);
        }
    }

    void correct(float* err, float* in, float* inerr = nullptr)
    {

        if (inerr)
        {
            memset(inerr, 0, inpNum * sizeof(float));
            for (int j = 0; j < neironsNum; j++)
            {
                 float* wptr = w + j * (inpNum + 1) + 1;
                 float grad =  der[j] * err[j];
                 for (int i = 0; i < inpNum; i++)
                     inerr[i] += grad * wptr[i];
            }
        }

        float s = getSpeed();
        b1t *= b1;
        b2t *= b2;

        float at = s * sqrt(1 - b2t) / (1 - b1t);


#pragma omp parallel for
        for (int j = 0; j < neironsNum; j++)
        {
            float* wptr = w + j * (inpNum + 1);
            float* mtptr = mt + j * (inpNum + 1);
            float* vtptr = vt + j * (inpNum + 1);

            float grad =  der[j] * err[j];
            ADAM_CORRECT(*wptr, s, *mtptr, *vtptr, grad);
            for (int i = 0; i < inpNum; i++)
                ADAM_CORRECT(wptr[i + 1], s, mtptr[i + 1], vtptr[i + 1], grad * in[i]);
        }
    }

    float*  getWeights()
    {
        return w;
    }
};

#define WX 32
#define WY 32

template <int inpNum, int middNum, int outNum>
class Network
{
    Layer<inpNum, middNum> inLayer;
    Layer<middNum, outNum> outLayer;
    float middleVals[middNum];
    float middleErr[middNum];
    float outErr[outNum];
public:
    Network():
        inLayer(sp), outLayer(sp)
    {}
    ~Network()
    {}
    void save(const char* name)
    {
       std::ofstream out(name);
       float* w = inLayer.getWeights();
       for (int i = 0; i < (inpNum + 1) * middNum; i++)
           out << w[i] << std::endl;

       w = outLayer.getWeights();
       for (int i = 0; i < (middNum + 1) * outNum; i++)
           out << w[i] << std::endl;

       out.close();
    }

    void drawWeights()
    {
         int cols = 9;
         int rows = 9;

         int Nx = (WX + 1) * cols + 1;
         int Ny = (WY + 1) * rows + 1;
         RGBImage outImg(Nx, Ny);

         outImg.fill(0.3, 0.3, 0.3);

         float* w = inLayer.getWeights() + 1;

         for (int i = 0; i < cols; i++)
             for (int j = 0; j < rows; j++)
             {
                 outImg.setRect(w, 1 + (WX + 1) * i, 1 + (WY + 1) * j, WX, WY);
                 w += inpNum + 1;
             }
         outImg.normalize();
         outImg.Save("color_w");

         w = inLayer.getWeights() + 1 + WX * WY * 3;

         for (int i = 0; i < cols; i++)
             for (int j = 0; j < rows; j++)
             {
                 outImg.setRect(w, 1 + (WX + 1) * i, 1 + (WY + 1) * j, WX, WY);
                 w += inpNum + 1;
             }
         outImg.normalize();
         outImg.Save("normals_w");

         w = inLayer.getWeights() + 1 + WX * WY * 3 * 2;

         for (int i = 0; i < cols; i++)
             for (int j = 0; j < rows; j++)
             {
                 outImg.setRect(w, 1 + (WX + 1) * i, 1 + (WY + 1) * j, WX, WY);
                 w += inpNum + 1;
             }
         outImg.normalize();
         outImg.Save("trans_w");
    }

    void load(const char* name)
    {
       std::ifstream in(name);
       float* w = inLayer.getWeights();
       for (int i = 0; i < (inpNum + 1) * middNum; i++)
             in >> w[i];

       w = outLayer.getWeights();
       for (int i = 0; i < (middNum + 1) * outNum; i++)
           in >> w[i];

       in.close();
    }

    float update(float* inp, float* ref, float* outp)
    {
     //   float dropouted[middNum];
     //   for (int i = 0; i < middNum; i++)
     //       dropouted[i] = rand() % 2;

        inLayer.calc(inp, middleVals);
      //  for (int i = 0; i < middNum; i++)
        //    middleVals[i] *= dropouted[i];
        outLayer.calc(middleVals, outp);

        float rmse = 0;
        for (int i = 0; i < outNum; i++)
        {
           outErr[i] = ref[i] - outp[i];
           rmse += outErr[i] * outErr[i];
        }

        outLayer.correct(outErr, middleVals, middleErr);
       // for (int i = 0; i < middNum; i++)
       //     middleErr[i] *= dropouted[i];
        inLayer.correct(middleErr, inp);

        return sqrt(rmse / inpNum);
    }
};

struct Sample
{
    RGBImage color;
    RGBImage normal;
    RGBImage trans;
    GRAYImage depth;
    RGBImage refImg;
    RGBImage filtered;
    char outFileName[128];

    Sample(const char* colorFile,
           const char* normalFile,
           const char* transFile,
           const char* depthFile,
           const char* refImgFile,
           const char* outFile) :
        color(colorFile), normal(normalFile),
        trans(transFile), depth(depthFile),
        refImg(refImgFile), filtered(colorFile)
    {
        color.normalize();
        normal.normalize();
        trans.normalize();
        depth.normalize();
        refImg.normalize();
        refImg.applyGamma(gamma);
        color.applyGamma(gamma);
        filtered.fill(0,0,0);

        strcpy(outFileName, outFile);
    }
    void fillInBuffer(float* buff, int x, int y, int w, int h)
    {
        color.getRect(buff, x - w / 2, y - h / 2, w, h);
        normal.getRect(buff + w * h * 3, x - w / 2, y - h / 2, w, h);
        trans.getRect(buff + w * h * 3 * 2, x - w / 2, y - h / 2, w, h);
        depth.getRect(buff + w * h * 3 * 3, x - w / 2, y - h / 2, w, h);
    }

    void flipHor()
    {
        color.flipHor();
        normal.flipHor();
        trans.flipHor();
        depth.flipHor();
        refImg.flipHor();
    }

    void flipVert()
    {
        color.flipVert();
        normal.flipVert();
        trans.flipVert();
        depth.flipVert();
        refImg.flipVert();
    }

};

void loadSamples(std::vector<Sample*>& s)
{
    for (int i = 1; i < 7; i++)
    {
       std::string folder = SAMPLES_FOLDER + "/s" + std::to_string(i) + "/";
       Sample* ptr = new Sample((folder + "color.exr").c_str(),
                                (folder + "normal.exr").c_str(),
                                (folder + "transmission.exr").c_str(),
                                (folder + "depth.exr").c_str(),
                                (folder + "1024samples.exr").c_str(),
                                (folder + "filtered").c_str());
       s.push_back(ptr);

       ptr = new Sample((folder + "color.exr").c_str(),
                                (folder + "normal.exr").c_str(),
                                (folder + "transmission.exr").c_str(),
                                (folder + "depth.exr").c_str(),
                                (folder + "1024samples.exr").c_str(),
                                (folder + "hf_filtered").c_str());
       ptr->flipHor();
       s.push_back(ptr);

       ptr = new Sample((folder + "color.exr").c_str(),
                                (folder + "normal.exr").c_str(),
                                (folder + "transmission.exr").c_str(),
                                (folder + "depth.exr").c_str(),
                                (folder + "1024samples.exr").c_str(),
                                (folder + "vf_filtered").c_str());
       ptr->flipVert();
       s.push_back(ptr);
    }
}

//0.000397871 3 * 8

int main()
{
    std::vector<Sample*> s;
    loadSamples(s);

    float inRect[WX * WY * 10];
    float out[3];
    float refRect[3];

    srand(time(NULL));
    Network<WX * WY * 10, 81, 3> encoder;

    int step = 0;
    float err = 0;

#if (LOAD == 1)
    encoder.load("w.txt");
#endif


    Sample* sptr = nullptr;

    int epoche_sizw = PERIOD;
    while (true)
    {
        step++;
        sptr = s[rand() % 18];

        int x = rand() % sptr->color.getW();
        int y = rand() % sptr->color.getH();

        sptr->fillInBuffer(inRect, x, y, WX, WY);
        sptr->refImg.getRect(refRect, x, y, 1, 1);

        err += encoder.update(inRect, refRect, out);
        sptr->filtered.setPixel(out, x, y);

        if (step % epoche_sizw == 0)
        {
            encoder.save("w.txt");
            encoder.drawWeights();
            sptr->filtered.Save(sptr->outFileName);
            std::cout << err / period  << std::endl;
            err = 0;
        }
    }
}
