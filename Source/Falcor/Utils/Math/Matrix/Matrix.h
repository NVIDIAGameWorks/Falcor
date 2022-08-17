/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

#include "Core/Assert.h"

#define GLM_FORCE_CTOR_INIT
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include "glm/packing.hpp"

#include <array>

namespace Falcor
{
/** The new matrix library (or wrapper for one).
    Name means "Row-Major, Column Vector" and signifies that the matrices are Row Major, and the vectors are column vectors (meaning M.v, translation in the last column)
 */
namespace rmcv
{
    template<int N, typename T = float>
    using vec = glm::vec<N, T, glm::defaultp>;

    using float1 = vec<1>;
    using float2 = vec<2>;
    using float3 = vec<3>;
    using float4 = vec<4>;

    using vec1 = vec<1>;
    using vec2 = vec<2>;
    using vec3 = vec<3>;
    using vec4 = vec<4>;

    template<int TRowCount, int TColCount, typename T>
    class matrix
    {
        // For now, we only support float, but want to have the type T visible
        static_assert(std::is_same_v<float, T>);

    private:
        template<int, int, typename>
        friend class matrix;
    public:
        using Type    = T;
        using RowType = vec<TColCount>;
        using ColType = vec<TRowCount>;

        static constexpr int kRowCount = TRowCount;
        static constexpr int kColCount = TColCount;

        static constexpr int getRowCount() { return TRowCount;  }
        static constexpr int getColCount() { return TColCount; }

        matrix(float diagonal = 1.f)
        {
            memset(this, 0, sizeof(*this));
            for (int i = 0; i < std::min(TRowCount, TColCount); ++i)
                mRows[i][i] = diagonal;
        }

        template<typename T>
        matrix(std::initializer_list<T> v)
        {
            float* f = &mRows[0][0];
            for (auto it = v.begin(); it != v.end(); ++it, ++f)
                *f = static_cast<float>(*it);
        }

        matrix(const matrix&) = default;
        matrix(matrix&&) noexcept = default;

        template<int R, int C>
        matrix(const matrix<R, C, T>& other) : matrix(1.f)
        {
            for (int r = 0; r < std::min(TRowCount, R); ++r)
            {
                memcpy(&mRows[r], &other.mRows[r], std::min(TColCount, C) * sizeof(T));
            }
        }

        matrix& operator=(const matrix&) = default;
        matrix& operator=(matrix&&) = default;

        template<int R, int C>
        matrix& operator=(const matrix<R, C, T>& other)
        {
            for (int r = 0; r < std::min(TRowCount, R); ++r)
            {
                memcpy(&mRows[r], &other.mRows[r], std::min(TColCount, C) * sizeof(float));
            }

            return *this;
        }

        float* data() { return &mRows[0][0]; }
        const float* data() const { return &mRows[0][0]; }

        RowType& operator[](unsigned r) { FALCOR_ASSERT_LT(r, TRowCount); return mRows[r]; }
        const RowType& operator[](unsigned r) const { FALCOR_ASSERT_LT(r, TRowCount); return mRows[r]; }

        // Doing this instead of [], since Falcor heavily uses [] on GLM to obtain the column
        //RowType& getRow(int row) { return mRows[row]; } // don't want write access even more than getters, for now

        //const RowType& getRow(int row) const { return mRows[row]; }

        //void setRow(int row, const RowType& v) { mRows[row] = v; }

        ColType getCol(int col) const
        {
            ColType result;
            for (int r = 0; r < TRowCount; ++r)
                result[r] = mRows[r][col];
            return result;
        }

        void setCol(int col, const ColType& v)
        {
            for (int r = 0; r < TRowCount; ++r)
                mRows[r][col] = v[r];
        }

        //RowType& operator[](int row) { return mRows[row]; }
        //const RowType& operator[](int row) const { return mRows[row]; }

        matrix<TColCount, TRowCount, T> getTranspose() const
        {
            matrix<TColCount, TRowCount, T> result;
            for (int r = 0; r < TRowCount; ++r)
            {
                for (int c = 0; c < TColCount; ++c)
                {
                    result.mRows[c][r] = mRows[r][c];
                }
            }

            return result;
        }

        bool operator==(const matrix& rhs) const
        {
            return memcmp(this, &rhs, sizeof(*this)) == 0;
        }

        bool operator!=(const matrix& rhs) const
        {
            return !(*this == rhs);
        }

        friend std::ostream& operator<<(std::ostream& os, const matrix& x)
        {
            os << "{";
            for (unsigned r = 0; r < x.kRowCount; ++r)
            {
                os << "{";
                for (unsigned c = 0; c < x.kColCount; ++c)
                {
                    if (c > 0)
                        os << ", ";
                    os << x[r][c];
                }
                os << "}";
            }
            os << "}";
            return os;
        }
    private:
        RowType mRows[TRowCount];
    };

    using mat2 = rmcv::matrix<2, 2, float>;
    using mat3 = rmcv::matrix<3, 3, float>;
    using mat4 = rmcv::matrix<4, 4, float>;

    using mat1x1 = rmcv::matrix<1, 1, float>;
    using mat2x1 = rmcv::matrix<2, 1, float>;
    using mat3x1 = rmcv::matrix<3, 1, float>;
    using mat4x1 = rmcv::matrix<4, 1, float>;

    using mat2x1 = rmcv::matrix<2, 1, float>;
    using mat2x2 = rmcv::matrix<2, 2, float>;
    using mat3x2 = rmcv::matrix<3, 2, float>;
    using mat4x2 = rmcv::matrix<4, 2, float>;

    using mat1x3 = rmcv::matrix<1, 3, float>;
    using mat2x3 = rmcv::matrix<2, 3, float>;
    using mat3x3 = rmcv::matrix<3, 3, float>;
    using mat4x3 = rmcv::matrix<4, 3, float>;

    using mat1x4 = rmcv::matrix<1, 4, float>;
    using mat2x4 = rmcv::matrix<2, 4, float>;
    using mat3x4 = rmcv::matrix<3, 4, float>;
    using mat4x4 = rmcv::matrix<4, 4, float>;

    template<int R, int C, typename T>
    glm::mat<C, R, T> toGLM(const matrix<R, C, T>& src)
    {
        glm::mat<C, R, T> result;
        for (int c = 0; c < C; ++c)
            result[c] = src.getCol(c);
        return result;
    }

    template<int R, int C, typename T>
    rmcv::matrix<R, C, T> toRMCV(const glm::mat<C, R, T>& src)
    {
        rmcv::matrix<R, C, T> result;
        for (int c = 0; c < C; ++c)
            result.setCol(c, src[c]);
        return result;
    }

    template<int R, int C, typename T>
    matrix<C, R, T> transpose(const matrix<R, C, T>& m)
    {
        return m.getTranspose();
    }

    template<typename T>
    matrix<4, 4, T> translate(const matrix<4, 4, T>& m, const vec<3, T>& v)
    {
        return toRMCV(glm::translate(toGLM(m), v));
    }

    template<typename T>
    matrix<4, 4, T> translate(const vec<3, T>& v)
    {
        return translate(matrix<4, 4, T>(T(1)), v);
    }


    template<typename T>
    matrix<4, 4, T> rotate(const matrix<4, 4, T>& m, T angle, const vec<3, T>& v)
    {
        return toRMCV(glm::rotate(toGLM(m), angle, v));
    }

    template<typename T>
    matrix<4, 4, T> rotate(T angle, const vec<3, T>& v)
    {
        return rotate(matrix<4, 4, T>(T(1)), angle, v);
    }

    template<typename T>
    matrix<4, 4, T> scale(const matrix<4, 4, T>& m, const vec<3, T>& v)
    {
        return toRMCV(glm::scale(toGLM(m), v));
    }

    template<typename T>
    matrix<4, 4, T> scale(const vec<3, T>& v)
    {
        return scale(matrix<4, 4, T>(T(1)), v);
    }


    template<int M, int N, typename T>
    matrix<M, N, T> inverse(const matrix<M, N, T>& m)
    {
        return toRMCV(glm::inverse(toGLM(m)));
    }

    template<typename Matrix>
    Matrix identity()
    {
        return Matrix(1.f);
    }

    template<int M, int N, typename T>
    T determinant(const matrix<M, N, T>& m)
    {
        return glm::determinant(toGLM(m));
    }

    template<int M, int N, int P, typename T>
    matrix<M, P, T> operator*(const matrix<M, N, T>& lhs, const matrix<N, P, T>& rhs)
    {
        return toRMCV(toGLM(lhs) * toGLM(rhs));

        // Causes precision difference
        //matrix<M, P> result(0.f);
        //for (int m = 0; m < M; ++m)
        //    for (int p = 0; p < P; ++p)
        //        result.setElement(m, p, glm::dot(lhs.getRow(m), rhs.getCol(p)));

        //return result;
    }

    template<int M, int N, typename T>
    matrix<M, N, T> operator*(const matrix<M, N, T>& lhs, const T& rhs)
    {
        return toRMCV(toGLM(lhs) * rhs);
    }

    template<int M, int N, typename T>
    vec<M, T> operator*(const matrix<M, N, T>& lhs, const vec<N, T>& rhs)
    {
        return toGLM(lhs) * rhs;
    }

    template<int M, int N, typename T>
    vec<N, T> operator*(const vec<M, T>& lhs, const matrix<M, N, T>& rhs)
    {
        return lhs * toGLM(rhs);
    }


    template<int M, int N, typename T>
    std::string to_string(const matrix<M, N, T>& m)
    {
        return glm::to_string(toGLM(m));
    }

    template<typename T>
    void extractEulerAngleXYZ(const matrix<4, 4, T>& m, float& t1, float& t2, float& t3)
    {
        glm::extractEulerAngleXYZ(toGLM(m), t1, t2, t3);
    }

    inline mat4 lookAtLH(const vec3& eye, const vec3& center, const vec3& up)
    {
        return toRMCV(glm::lookAtLH(eye, center, up));
    }

    // GLM decides whether to call LH or RH based on its
    inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
    {
        return toRMCV(glm::lookAt(eye, center, up));
    }

    inline mat4 ortho(float left, float right, float bottom, float top, float zNear, float zFar)
    {
        return toRMCV(glm::ortho(left, right, bottom, top, zNear, zFar));
    }

    inline mat4 perspective(float fovy, float aspect, float zNear, float zFar)
    {
        return toRMCV(glm::perspective(fovy, aspect, zNear, zFar));
    }

    inline mat4 eulerAngleY(float radians)
    {
        return toRMCV(glm::eulerAngleY(radians));
    }

    inline mat4 eulerAngleX(float radians)
    {
        return toRMCV(glm::eulerAngleX(radians));
    }

    inline mat4 eulerAngleXYZ(float radiansX, float radiansY, float radiansZ)
    {
        return toRMCV(glm::eulerAngleXYZ(radiansX, radiansY, radiansZ));
    }

    inline mat4 mat4_cast(const glm::quat& quat)
    {
        return toRMCV(glm::mat4_cast(quat));
    }

    inline mat3 mat3_cast(const glm::quat& quat)
    {
        return toRMCV(glm::mat3_cast(quat));
    }

    inline mat3 make_mat3(const float* rowMajor)
    {
        mat3 result;
        memcpy(&result, rowMajor, sizeof(float) * 9);
        return result;
    }

    inline mat4 make_mat4(const float* rowMajor)
    {
        mat4 result;
        memcpy(&result, rowMajor, sizeof(float) * 16);
        return result;
    }

    inline mat3 make_mat3_fromCols(float3 col0, float3 col1, float3 col2)
    {
        mat3 result;
        result[0] = col0;
        result[1] = col1;
        result[2] = col2;
        return result.getTranspose();
    }

    inline bool decompose(mat4 const& modelMatrix, vec3& scale, glm::quat& orientation, vec3& translation, vec3& skew, vec4& perspective)
    {
        return glm::decompose(toGLM(modelMatrix), scale, orientation, translation, skew, perspective);
    }

    inline mat3 diagonal3x3(float3 scale)
    {
        return toRMCV(glm::diagonal3x3(scale));
    }

    template<int R, int C, typename T>
    bool lex_lt(const matrix<R, C, T>& lhs, const matrix<R, C, T>& rhs)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int c = 0; c < C; ++c)
            {
                if (lhs[r][c] != rhs[r][c])
                    return lhs[r][c] < rhs[r][c];
            }
        }
        return false;
    }
}

};
