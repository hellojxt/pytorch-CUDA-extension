#pragma once
#include "array.h"
namespace diffRender
{

	template <typename T>
	class GArr2D
	{
	public:
		GArr<T> data;
		int rows, cols;
		void resize(int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data.resize(rows * cols);
		}
		GArr2D() {}
		GArr2D(int rows, int cols)
		{
			this->resize(rows, cols);
		}

		GArr2D(T *data, int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data = GArr<T>(data, rows * cols);
		}

		void clear()
		{
			data.clear();
		}
		void reset()
		{
			data.reset();
		}

		~GArr2D(){};

		GPU_FUNC inline const T &operator()(const uint i, const uint j) const
		{
			return data[i * cols + j];
		}

		GPU_FUNC inline T &operator()(const uint i, const uint j)
		{
			return data[i * cols + j];
		}

		CGPU_FUNC inline int index(const uint i, const uint j) const
		{
			return i * cols + j;
		}

		inline GArr<T> operator[](unsigned int id)
		{
			return GArr<T>(begin() + id * cols, cols);
		}

		inline const GArr<T> operator[](unsigned int id) const
		{
			return GArr<T>(begin() + id * cols, cols);
		}

		CGPU_FUNC inline const T *begin() const { return data.begin(); }
		CGPU_FUNC inline T *begin() { return data.begin(); }
	};
}