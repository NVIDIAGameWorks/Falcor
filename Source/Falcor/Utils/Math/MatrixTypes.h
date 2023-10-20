/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once

#include "ScalarTypes.h"
#include "VectorTypes.h"
#include "Core/Error.h"

#include <limits>

namespace Falcor
{
namespace math
{

/**
 * Matrix type with row-major storage.
 *
 * The semantics are aligned with Slang:
 * - Row major storage
 * - Math operators are element-wise (e.g. +, -, *, /)
 * - Free standing functions for matrix operations (e.g. mul(), transpose(), etc.)
 *
 * @tparam T Scalar type
 * @tparam RowCount Number of rows (1-4)
 * @tparam ColCount Number of columns (1-4)
 */
template<typename T, int RowCount, int ColCount>
class matrix
{
    static_assert(RowCount >= 1 && RowCount <= 4);
    static_assert(ColCount >= 1 && ColCount <= 4);

    // For now, we only support float, but want to have the type T visible
    static_assert(std::is_same_v<float, T>);

private:
    template<typename, int, int>
    friend class matrix;

public:
    using value_type = T;
    using RowType = vector<T, ColCount>;
    using ColType = vector<T, RowCount>;

    static constexpr int getRowCount() { return RowCount; }
    static constexpr int getColCount() { return ColCount; }

    matrix() : matrix(Form::Identity) {}

    template<typename U>
    matrix(std::initializer_list<U> v)
    {
        T* f = &mRows[0][0];
        for (auto it = v.begin(); it != v.end(); ++it, ++f)
            *f = static_cast<T>(*it);
    }

    matrix(const matrix&) = default;
    matrix(matrix&&) noexcept = default;

    /// Construct matrix from another matrix with different dimensions.
    /// In HLSL/Slang, destination matrix must be equal or smaller than source matrix.
    /// In Falcor, destination matrix can be larger than source matrix (initialized with identity).
    template<int R, int C>
    matrix(const matrix<T, R, C>& other) : matrix(Form::Identity)
    {
        for (int r = 0; r < std::min(RowCount, R); ++r)
        {
            std::memcpy(&mRows[r], &other.mRows[r], std::min(ColCount, C) * sizeof(T));
        }
    }

    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&) = default;

    template<int R, int C>
    matrix& operator=(const matrix<T, R, C>& other)
    {
        for (int r = 0; r < std::min(RowCount, R); ++r)
        {
            std::memcpy(&mRows[r], &other.mRows[r], std::min(ColCount, C) * sizeof(T));
        }

        return *this;
    }

    /// Zero matrix.
    [[nodiscard]] static matrix zeros() { return matrix(Form::Zeros); }

    /// Identity matrix.
    [[nodiscard]] static matrix identity() { return matrix(Form::Identity); }

    T* data() { return &mRows[0][0]; }
    const T* data() const { return &mRows[0][0]; }

    RowType& operator[](int r)
    {
        FALCOR_ASSERT_LT(r, RowCount);
        return mRows[r];
    }
    const RowType& operator[](int r) const
    {
        FALCOR_ASSERT_LT(r, RowCount);
        return mRows[r];
    }

    RowType& getRow(int r)
    {
        FALCOR_ASSERT_LT(r, RowCount);
        return mRows[r];
    }
    const RowType& getRow(int r) const
    {
        FALCOR_ASSERT_LT(r, RowCount);
        return mRows[r];
    }

    void setRow(int r, const RowType& v)
    {
        FALCOR_ASSERT_LT(r, RowCount);
        mRows[r] = v;
    }

    ColType getCol(int col) const
    {
        FALCOR_ASSERT_LT(col, ColCount);
        ColType result;
        for (int r = 0; r < RowCount; ++r)
            result[r] = mRows[r][col];
        return result;
    }

    void setCol(int col, const ColType& v)
    {
        FALCOR_ASSERT_LT(col, ColCount);
        for (int r = 0; r < RowCount; ++r)
            mRows[r][col] = v[r];
    }

    bool operator==(const matrix& rhs) const { return std::memcmp(this, &rhs, sizeof(*this)) == 0; }
    bool operator!=(const matrix& rhs) const { return !(*this == rhs); }

private:
    enum class Form
    {
        Undefined,
        Zeros,
        Identity,
    };

    explicit matrix(Form form)
    {
        switch (form)
        {
        case Form::Undefined:
#ifdef _DEBUG
            for (int i = 0; i < RowCount; ++i)
                mRows[i] = RowType(std::numeric_limits<T>::quiet_NaN());
#endif
            break;
        case Form::Zeros:
            std::memset(this, 0, sizeof(*this));
            break;
        case Form::Identity:
            std::memset(this, 0, sizeof(*this));
            for (int i = 0; i < std::min(RowCount, ColCount); ++i)
                mRows[i][i] = T(1);
            break;
        }
    }

    RowType mRows[RowCount];
};

using float2x2 = matrix<float, 2, 2>;

using float3x3 = matrix<float, 3, 3>;

using float1x4 = matrix<float, 1, 4>;
using float2x4 = matrix<float, 2, 4>;
using float3x4 = matrix<float, 3, 4>;
using float4x4 = matrix<float, 4, 4>;

} // namespace math

using float2x2 = math::float2x2;

using float3x3 = math::float3x3;

using float1x4 = math::float1x4;
using float2x4 = math::float2x4;
using float3x4 = math::float3x4;
using float4x4 = math::float4x4;

} // namespace Falcor
