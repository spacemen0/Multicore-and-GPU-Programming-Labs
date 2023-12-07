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

#define DIM 512

#define MYFLOAT float

unsigned char *d_pixels = NULL;
unsigned char *h_pixels = NULL;
int gImageWidth, gImageHeight;

int maxiter = 20;
MYFLOAT offsetx = -200, offsety = 0, zoom = 0;
MYFLOAT scale = 1.5;

struct cuComplex
{
    MYFLOAT r;
    MYFLOAT i;

    cuComplex(MYFLOAT a, MYFLOAT b) : r(a), i(b) {}

    float magnitude2(void)
    {
        return r * r + i * i;
    }

    cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int mandelbrot(MYFLOAT jx, MYFLOAT jy, int maxiter)
{
    cuComplex c(jx, jy);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < maxiter; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i;
    }

    return i;
}

__global__ void computeFractal(unsigned char *ptr, int maxiter, MYFLOAT offsetx, MYFLOAT offsety, MYFLOAT scale, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int offset = x + y * width;

        MYFLOAT jx = scale * (MYFLOAT)(width / 2 - x + offsetx / scale) / (width / 2);
        MYFLOAT jy = scale * (MYFLOAT)(height / 2 - y + offsety / scale) / (width / 2);

        // now calculate the value at that position
        int fractalValue = mandelbrot(jx, jy, maxiter);

        // Colorize it
        int red = 255 * fractalValue / maxiter;
        if (red > 255)
            red = 255 - red;
        int green = 255 * fractalValue * 4 / maxiter;
        if (green > 255)
            green = 255 - green;
        int blue = 255 * fractalValue * 20 / maxiter;
        if (blue > 255)
            blue = 255 - blue;

        ptr[offset * 4 + 0] = red;
        ptr[offset * 4 + 1] = green;
        ptr[offset * 4 + 2] = blue;

        ptr[offset * 4 + 3] = 255;
    }
}

void print_help(){

};
void PrintHelp(){

};
void initBitmap(int width, int height)
{
    if (h_pixels)
        free(h_pixels);
    h_pixels = (unsigned char *)malloc(width * height * 4);
    gImageWidth = width;
    gImageHeight = height;
}

void Reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    glLoadIdentity();
    glOrtho(-0.5f, width - 0.5f, -0.5f, height - 0.5f, -1.f, 1.f);
    initBitmap(width, height);

    glutPostRedisplay();
}

int mouse_x, mouse_y, mouse_btn;

void mouse_button(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        // Record start position
        mouse_x = x;
        mouse_y = y;
        mouse_btn = button;
    }
}

void mouse_motion(int x, int y)
{
    if (mouse_btn == 0)
    {
        // Ordinary mouse button - move
        offsetx += (x - mouse_x) * scale;
        mouse_x = x;
        offsety += (mouse_y - y) * scale;
        mouse_y = y;

        glutPostRedisplay();
    }
    else
    {
        // Alt mouse button - scale
        scale *= pow(1.1, y - mouse_y);
        mouse_y = y;
        glutPostRedisplay();
    }
}

void KeyboardProc(unsigned char key, int x, int y)
{
    switch (key)
    {
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



void Draw()
{
    dim3 blockDim(16, 16);
    dim3 gridDim((gImageWidth + blockDim.x - 1) / blockDim.x, (gImageHeight + blockDim.y - 1) / blockDim.y);

    computeFractal<<<gridDim, blockDim>>>(d_pixels, maxiter, offsetx, offsety, scale, gImageWidth, gImageHeight);
    cudaMemcpy(h_pixels, d_pixels, gImageWidth * gImageHeight * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Dump the whole picture onto the screen. (Old-style OpenGL but without lots of geometry that doesn't matter so much.)
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(gImageWidth, gImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, h_pixels);

    PrintHelp();

    glutSwapBuffers();
}

int main(int argc, char **argv)
{
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
