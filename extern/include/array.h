#pragma once
#include <cassert>
#include <vector>
#include <iostream>
#include <memory>
#include "macro.h"

namespace diffRender
{

	/*!
	 *	\class	Array
	 *	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	 */
	template <typename T>
	class GArr
	{
	public:
		GArr(){};

		GArr(uint num)
		{
			this->resize(num);
		}

		GArr(T *data_, uint num)
		{
			this->m_data = data_;
			this->m_totalNum = num;
		}

		/*!
		 *	\brief	Do not release memory here, call clear() explicitly.
		 */
		~GArr(){};

		void resize(const uint n)
		{
			//		assert(n >= 1);
			if (m_data != nullptr)
				clear();

			m_totalNum = n;
			if (n == 0)
			{
				m_data = nullptr;
			}
			else
				cuSafeCall(cudaMalloc(&m_data, n * sizeof(T)));
		}

		/*!
		 *	\brief	Clear all data to zero.
		 */
		void reset()
		{
			cuSafeCall(cudaMemset((void *)m_data, 0, m_totalNum * sizeof(T)));
		}

		/*!
		 *	\brief	Free allocated memory.	Should be called before the object is deleted.
		 */
		void clear()
		{
			if (m_data != NULL)
			{
				cuSafeCall(cudaFree((void *)m_data));
			}

			m_data = NULL;
			m_totalNum = 0;
		}

		CGPU_FUNC inline const T *begin() const { return m_data; }
		CGPU_FUNC inline T *begin() { return m_data; }

		CGPU_FUNC inline const T *data() const { return m_data; }
		CGPU_FUNC inline T *data() { return m_data; }

		CGPU_FUNC inline const T *end() const { return m_data + m_totalNum; }
		CGPU_FUNC inline T *end() { return m_data + m_totalNum; }

		GPU_FUNC inline T &operator[](unsigned int id)
		{
			return m_data[id];
		}

		GPU_FUNC inline T &operator[](unsigned int id) const
		{
			return m_data[id];
		}

		CGPU_FUNC inline uint size() const { return m_totalNum; }

	private:
		T *m_data = 0;
		uint m_totalNum = 0;
	};
}