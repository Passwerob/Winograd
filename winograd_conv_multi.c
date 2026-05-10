// winograd_conv_multi.c
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define H 514
#define W 514
#define OH (H - 2)
#define OW (W - 2)

#define OUT_CH 128
#define REPEAT 3

static double now_sec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void fill_data(float *input, float *kernel)
{
    for (int i = 0; i < H * W; i++)
    {
        input[i] = (float)((i * 13 + 7) % 256) / 255.0f;
    }

    for (int oc = 0; oc < OUT_CH; oc++)
    {
        for (int k = 0; k < 9; k++)
        {
            kernel[oc * 9 + k] =
                (float)(((oc * 17 + k * 5 + 3) % 11) - 5) / 11.0f;
        }
    }
}

// 原始 3x3 卷积
// 输入:  单通道 H x W
// 卷积核: OUT_CH 个 3x3 kernel
// 输出:  OH x OW x OUT_CH
static void conv3x3_naive_multi(const float *restrict input,
                                const float *restrict kernel,
                                float *restrict output)
{
    for (int y = 0; y < OH; y++)
    {
        for (int x = 0; x < OW; x++)
        {
            float d00 = input[(y + 0) * W + (x + 0)];
            float d01 = input[(y + 0) * W + (x + 1)];
            float d02 = input[(y + 0) * W + (x + 2)];

            float d10 = input[(y + 1) * W + (x + 0)];
            float d11 = input[(y + 1) * W + (x + 1)];
            float d12 = input[(y + 1) * W + (x + 2)];

            float d20 = input[(y + 2) * W + (x + 0)];
            float d21 = input[(y + 2) * W + (x + 1)];
            float d22 = input[(y + 2) * W + (x + 2)];

            int out_base = (y * OW + x) * OUT_CH;

            for (int oc = 0; oc < OUT_CH; oc++)
            {
                const float *k = kernel + oc * 9;

                output[out_base + oc] =
                    d00 * k[0] + d01 * k[1] + d02 * k[2] +
                    d10 * k[3] + d11 * k[4] + d12 * k[5] +
                    d20 * k[6] + d21 * k[7] + d22 * k[8];
            }
        }
    }
}

// Winograd 卷积核变换
// U = G * g * G^T
static void transform_kernel_winograd(const float *restrict kernel,
                                      float *restrict U)
{
    for (int oc = 0; oc < OUT_CH; oc++)
    {
        const float *g = kernel + oc * 9;
        float *u = U + oc * 16;

        float g00 = g[0], g01 = g[1], g02 = g[2];
        float g10 = g[3], g11 = g[4], g12 = g[5];
        float g20 = g[6], g21 = g[7], g22 = g[8];

        float t00 = g00;
        float t01 = g01;
        float t02 = g02;

        float t10 = 0.5f * (g00 + g10 + g20);
        float t11 = 0.5f * (g01 + g11 + g21);
        float t12 = 0.5f * (g02 + g12 + g22);

        float t20 = 0.5f * (g00 - g10 + g20);
        float t21 = 0.5f * (g01 - g11 + g21);
        float t22 = 0.5f * (g02 - g12 + g22);

        float t30 = g20;
        float t31 = g21;
        float t32 = g22;

        u[0]  = t00;
        u[1]  = 0.5f * (t00 + t01 + t02);
        u[2]  = 0.5f * (t00 - t01 + t02);
        u[3]  = t02;

        u[4]  = t10;
        u[5]  = 0.5f * (t10 + t11 + t12);
        u[6]  = 0.5f * (t10 - t11 + t12);
        u[7]  = t12;

        u[8]  = t20;
        u[9]  = 0.5f * (t20 + t21 + t22);
        u[10] = 0.5f * (t20 - t21 + t22);
        u[11] = t22;

        u[12] = t30;
        u[13] = 0.5f * (t30 + t31 + t32);
        u[14] = 0.5f * (t30 - t31 + t32);
        u[15] = t32;
    }
}

// Winograd F(2x2, 3x3)
// 每个 4x4 输入块产生 2x2 输出块
static void conv3x3_winograd_multi(const float *restrict input,
                                   const float *restrict U,
                                   float *restrict output)
{
    for (int y = 0; y < OH; y += 2)
    {
        for (int x = 0; x < OW; x += 2)
        {
            float d00 = input[(y + 0) * W + (x + 0)];
            float d01 = input[(y + 0) * W + (x + 1)];
            float d02 = input[(y + 0) * W + (x + 2)];
            float d03 = input[(y + 0) * W + (x + 3)];

            float d10 = input[(y + 1) * W + (x + 0)];
            float d11 = input[(y + 1) * W + (x + 1)];
            float d12 = input[(y + 1) * W + (x + 2)];
            float d13 = input[(y + 1) * W + (x + 3)];

            float d20 = input[(y + 2) * W + (x + 0)];
            float d21 = input[(y + 2) * W + (x + 1)];
            float d22 = input[(y + 2) * W + (x + 2)];
            float d23 = input[(y + 2) * W + (x + 3)];

            float d30 = input[(y + 3) * W + (x + 0)];
            float d31 = input[(y + 3) * W + (x + 1)];
            float d32 = input[(y + 3) * W + (x + 2)];
            float d33 = input[(y + 3) * W + (x + 3)];

            // V = B^T * d * B
            float t00 = d00 - d20;
            float t01 = d01 - d21;
            float t02 = d02 - d22;
            float t03 = d03 - d23;

            float t10 = d10 + d20;
            float t11 = d11 + d21;
            float t12 = d12 + d22;
            float t13 = d13 + d23;

            float t20 = -d10 + d20;
            float t21 = -d11 + d21;
            float t22 = -d12 + d22;
            float t23 = -d13 + d23;

            float t30 = d10 - d30;
            float t31 = d11 - d31;
            float t32 = d12 - d32;
            float t33 = d13 - d33;

            float V0  = t00 - t02;
            float V1  = t01 + t02;
            float V2  = -t01 + t02;
            float V3  = t01 - t03;

            float V4  = t10 - t12;
            float V5  = t11 + t12;
            float V6  = -t11 + t12;
            float V7  = t11 - t13;

            float V8  = t20 - t22;
            float V9  = t21 + t22;
            float V10 = -t21 + t22;
            float V11 = t21 - t23;

            float V12 = t30 - t32;
            float V13 = t31 + t32;
            float V14 = -t31 + t32;
            float V15 = t31 - t33;

            int out00 = ((y + 0) * OW + (x + 0)) * OUT_CH;
            int out01 = ((y + 0) * OW + (x + 1)) * OUT_CH;
            int out10 = ((y + 1) * OW + (x + 0)) * OUT_CH;
            int out11 = ((y + 1) * OW + (x + 1)) * OUT_CH;

            for (int oc = 0; oc < OUT_CH; oc++)
            {
                const float *u = U + oc * 16;

                float M0  = u[0]  * V0;
                float M1  = u[1]  * V1;
                float M2  = u[2]  * V2;
                float M3  = u[3]  * V3;

                float M4  = u[4]  * V4;
                float M5  = u[5]  * V5;
                float M6  = u[6]  * V6;
                float M7  = u[7]  * V7;

                float M8  = u[8]  * V8;
                float M9  = u[9]  * V9;
                float M10 = u[10] * V10;
                float M11 = u[11] * V11;

                float M12 = u[12] * V12;
                float M13 = u[13] * V13;
                float M14 = u[14] * V14;
                float M15 = u[15] * V15;

                // Y = A^T * M * A
                float r00 = M0 + M4 + M8;
                float r01 = M1 + M5 + M9;
                float r02 = M2 + M6 + M10;
                float r03 = M3 + M7 + M11;

                float r10 = M4 - M8 - M12;
                float r11 = M5 - M9 - M13;
                float r12 = M6 - M10 - M14;
                float r13 = M7 - M11 - M15;

                output[out00 + oc] = r00 + r01 + r02;
                output[out01 + oc] = r01 - r02 - r03;
                output[out10 + oc] = r10 + r11 + r12;
                output[out11 + oc] = r11 - r12 - r13;
            }
        }
    }
}

static float max_abs_error(const float *a, const float *b)
{
    float max_err = 0.0f;
    int total = OH * OW * OUT_CH;

    for (int i = 0; i < total; i++)
    {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err)
        {
            max_err = err;
        }
    }

    return max_err;
}

int main(void)
{
    float *input = NULL;
    float *kernel = NULL;
    float *U = NULL;
    float *out_naive = NULL;
    float *out_winograd = NULL;

    posix_memalign((void **)&input, 64, H * W * sizeof(float));
    posix_memalign((void **)&kernel, 64, OUT_CH * 9 * sizeof(float));
    posix_memalign((void **)&U, 64, OUT_CH * 16 * sizeof(float));
    posix_memalign((void **)&out_naive, 64, OH * OW * OUT_CH * sizeof(float));
    posix_memalign((void **)&out_winograd, 64, OH * OW * OUT_CH * sizeof(float));

    if (!input || !kernel || !U || !out_naive || !out_winograd)
    {
        printf("Memory allocation failed.\n");
        free(input);
        free(kernel);
        free(U);
        free(out_naive);
        free(out_winograd);
        return 1;
    }

    fill_data(input, kernel);
    transform_kernel_winograd(kernel, U);

    // 预热
    conv3x3_naive_multi(input, kernel, out_naive);
    conv3x3_winograd_multi(input, U, out_winograd);

    double t0, t1;
    double naive_time;
    double winograd_time;

    t0 = now_sec();
    for (int r = 0; r < REPEAT; r++)
    {
        conv3x3_naive_multi(input, kernel, out_naive);
    }
    t1 = now_sec();
    naive_time = (t1 - t0) / REPEAT;

    t0 = now_sec();
    for (int r = 0; r < REPEAT; r++)
    {
        conv3x3_winograd_multi(input, U, out_winograd);
    }
    t1 = now_sec();
    winograd_time = (t1 - t0) / REPEAT;

    float err = max_abs_error(out_naive, out_winograd);

    printf("Input size: %d x %d\n", H, W);
    printf("Output size: %d x %d\n", OH, OW);
    printf("Output channels: %d\n", OUT_CH);
    printf("Kernel size: 3 x 3\n");
    printf("Repeat: %d\n", REPEAT);
    printf("Naive 3x3 convolution avg time:      %.6f s\n", naive_time);
    printf("Winograd F(2x2,3x3) avg time:        %.6f s\n", winograd_time);
    printf("Speedup: %.3fx\n", naive_time / winograd_time);
    printf("Max abs error: %.8f\n", err);

    free(input);
    free(kernel);
    free(U);
    free(out_naive);
    free(out_winograd);

    return 0;
}
