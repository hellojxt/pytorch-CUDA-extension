#pragma once
#include "array2D.h"
namespace diffRender
{

    template <typename T>
    class GArr3D
    {
    public:
        GArr<T> data;
        int batchs, rows, cols;
        void resize(int batchs, int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->batchs = batchs;
            this->data.resize(rows * cols * batchs);
        }
        GArr3D() {}
        GArr3D(int batchs, int rows, int cols)
        {
            this->resize(batchs, rows, cols);
        }
        GArr3D(T *data, int batchs, int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->batchs = batchs;
            this->data = GArr<T>(data, rows * cols * batchs);
        }
        void clear()
        {
            data.clear();
        }
        void reset()
        {
            data.reset();
        }

        ~GArr3D(){};
        GPU_FUNC inline const T &operator()(const uint b_i, const uint i, const uint j) const
        {
            return data[b_i * rows * cols + i * cols + j];
        }

        GPU_FUNC inline T &operator()(const uint b_i, const uint i, const uint j)
        {
            return data[b_i * rows * cols + i * cols + j];
        }

        CGPU_FUNC inline int index(const uint b_i, const uint i, const uint j) const
        {
            return b_i * rows * cols + i * cols + j;
        }

        inline GArr2D<T> operator[](unsigned int id)
        {
            return GArr2D<T>(data.data() + id * rows * cols, rows, cols);
        }

        inline const GArr2D<T> operator[](unsigned int id) const
        {
            return GArr2D<T>(data.data() + id * rows * cols, rows, cols);
        }

        CGPU_FUNC inline const T *begin() const { return data.begin(); }
        CGPU_FUNC inline const T *end() const { return data.end(); }
        CGPU_FUNC inline T *begin() { return data.begin(); }
        CGPU_FUNC inline T *end() { return data.end(); }
    };
}