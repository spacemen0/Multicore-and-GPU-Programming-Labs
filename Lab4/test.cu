#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DIM 512

#define MYFLOAT float

unsigned char *d_pixels = NULL;
unsigned char *h_pixels = NULL;
int gImageWidth, gImageHeight;

int maxiter = 20;
MYFLOAT offsetx = -200, offsety = 0, zoom = 0;
MYFLOAT scale = 1.5;

struct cuComplex {
    MYFLOAT r;
    MYFLOAT i;

    __device__ cuComplex operator*(const cuComplex &a) const {
        return {r * a.r - i * a.i, r * a.i + i * a.r};
    }

    __device__ cuComplex operator+(const cuComplex &a) const {
        return {r + a.r, i + a.i};
    }

    __device__ MYFLOAT magnitude2() const {
        return r * r + i * i;
    }
};

__device__ cuComplex make_cuComplex(MYFLOAT r, MYFLOAT i) {
    cuComplex c;
    c.r = r;
    c.i = i;
    return c;
}

__global__ void mandelbrotKernel(unsigned char *ptr, int gImageWidth, int gImageHeight,
                                 MYFLOAT offsetx, MYFLOAT offsety, MYFLOAT scale, int maxiter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < gImageWidth && y < gImageHeight) {
        int offset = x + y * gImageWidth;

        MYFLOAT jx = scale * (MYFLOAT)(gImageWidth / 2 - x + offsetx / scale) / (gImageWidth / 2);
        MYFLOAT jy = scale * (MYFLOAT)(gImageHeight / 2 - y + offsety / scale) / (gImageWidth / 2);

        cuComplex c = make_cuComplex(jx, jy);
        cuComplex a = make_cuComplex(jx, jy);

        int i = 0;
        for (i = 0; i < maxiter; i++) {
            a = a * a + c;
            if (a.magnitude2() > 1000)
                break;
        }

        // Colorize it
        int red = 255 * i / maxiter;
        if (red > 255)
            red = 255 - red;
        int green = 255 * i * 4 / maxiter;
        if (green > 255)
            green = 255 - green;
        int blue = 255 * i * 20 / maxiter;
        if (blue > 255)
            blue = 255 - blue;

        ptr[offset * 4 + 0] = red;
        ptr[offset * 4 + 1] = green;
        ptr[offset * 4 + 2] = blue;
        ptr[offset * 4 + 3] = 255;
    }
}

void launchKernel() {
    const int numThreadsPerBlock = 16;
    dim3 threadsPerBlock(numThreadsPerBlock, numThreadsPerBlock);
    dim3 numBlocks((gImageWidth + numThreadsPerBlock - 1) / numThreadsPerBlock,
                   (gImageHeight + numThreadsPerBlock - 1) / numThreadsPerBlock);

    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_pixels, gImageWidth, gImageHeight, offsetx, offsety, scale, maxiter);
    cudaDeviceSynchronize();  // Ensure the kernel is completed before copying back the data
}

void initBitmap(int width, int height) {
    if (h_pixels)
        free(h_pixels);
    gImageWidth = width;
    gImageHeight = height;
}

void Reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glLoadIdentity();
    glOrtho(-0.5f, width - 0.5f, -0.5f, height - 0.5f, -1.f, 1.f);
    initBitmap(width, height);

    glutPostRedisplay();
}

int mouse_x, mouse_y, mouse_btn;

void mouse_button(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        // Record start position
        mouse_x = x;
        mouse_y = y;
        mouse_btn = button;
    }
}

void mouse_motion(int x, int y) {
    if (mouse_btn == 0) {
        // Ordinary mouse button - move
        offsetx += (x - mouse_x) * scale;
        mouse_x = x;
        offsety += (mouse_y - y) * scale;
        mouse_y = y;

        glutPostRedisplay();
    } else {
        // Alt mouse button - scale
        scale *= pow(1.1, y - mouse_y);
        mouse_y = y;
        glutPostRedisplay();
    }
}

void KeyboardProc(unsigned char key, int x, int y) {
    switch (key) {
    case 27: /* Escape key */
    case 'q':
    case 'Q':
        exit(0);
        break;
    case '+':
        maxiter += maxiter < 1024 - 32 ? 32 : 0;
        break;
    case '-':
        maxiter -= maxiter > 0 + 32 ? 32 : 0;
        break;
    // case 'h':
    //     print_help = !print_help;
    //     break;
    }
    glutPostRedisplay();
}

void Draw() {
    dim3 blockDim(16, 16);
    dim3 gridDim((gImageWidth + blockDim.x - 1) / blockDim.x, (gImageHeight + blockDim.y - 1) / blockDim.y);

    launchKernel();
    cudaMemcpy(h_pixels, d_pixels, gImageWidth * gImageHeight * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Dump the whole picture onto the screen.
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(gImageWidth, gImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, h_pixels);

    glutSwapBuffers();
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow("Mandelbrot explorer (CUDA)");

    // Allocate memory on the host and device
    h_pixels = (unsigned char *)malloc(DIM * DIM * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&d_pixels, DIM * DIM * 4 * sizeof(unsigned char));

    glutDisplayFunc(Draw);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    glutKeyboardFunc(KeyboardProc);
    glutReshapeFunc(Reshape);

    initBitmap(DIM, DIM);

    glutMainLoop();

    // Free allocated memory
    free(h_pixels);
    cudaFree(d_pixels);

    return 0;
}
