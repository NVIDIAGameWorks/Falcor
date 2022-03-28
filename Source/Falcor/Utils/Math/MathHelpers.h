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
#include "glm/gtc/quaternion.hpp"
#include "glm/geometric.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Falcor
{
    /** Generate a vector that is orthogonal to the input vector
        This can be used to invent a tangent frame for meshes that don't have real tangents/bitangents.
        \param[in] u Unit vector.
        \return v Unit vector that is orthogonal to u.
    */
    inline float3 perp_stark(const float3& u)
    {
        // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
        float3 a = abs(u);
        uint32_t uyx = (a.x - a.y) < 0 ? 1 : 0;
        uint32_t uzx = (a.x - a.z) < 0 ? 1 : 0;
        uint32_t uzy = (a.y - a.z) < 0 ? 1 : 0;
        uint32_t xm = uyx & uzx;
        uint32_t ym = (1 ^ xm) & uzy;
        uint32_t zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
        float3 v = normalize(cross(u, float3(xm, ym, zm)));
        return v;
    }

    /** Builds a local frame from a unit normal vector.
        \param[in] n Unit normal vector.
        \param[out] t Unit tangent vector.
        \param[out] b Unit bitangent vector.
    */
    inline void buildFrame(const float3& n, float3& t, float3& b)
    {
        t = perp_stark(n);
        b = cross(n, t);
    }

    /** Check if the specified matrix has no inf or nan values.
        \param[in] matrix The matrix to check.
        \return True if valid else false.
    */
    template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
    inline bool isMatrixValid(const glm::mat<C, R, T, Q>& matrix)
    {
        for (glm::length_t c = 0; c < C; c++)
        {
            if (glm::any(glm::isinf(matrix[c])) || glm::any(glm::isnan(matrix[c])))
                return false;
        }
        return true;
    }

    /** Check if the specified matrix is affine.
        \param[in] matrix The matrix to check.
        \return True if affine else false.
    */
    template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
    inline bool isMatrixAffine(const glm::mat<C, R, T, Q>& matrix)
    {
        GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_GENTYPE, "'isMatrixAffine' only accept floating-point inputs");

        const glm::length_t lastRow = R - 1;
        const glm::length_t lastCol = C - 1;

        bool affine = true;
        for (glm::length_t c = 0; c < lastCol; c++)
        {
            if (matrix[c][lastRow] != 0.f)
                affine = false;
        }

        if (matrix[lastCol][lastRow] != 1.f)
            affine = false;

        return affine;
    }

    /** Check if transform matrix have no inf/nan values and if it is affine. If it is not affine, it will return an affine matrix and if it is not valid, it will throw a runtime error.
        \param[in] transform Transform matrix.
        \return A copy of the matrix that is affine.
    */
    inline glm::mat4x4 validateTransformMatrix(const glm::mat4x4& transform)
    {
        glm::mat4x4 newMatrix(transform);

        if (!isMatrixValid(newMatrix))
        {
            throw RuntimeError("Transform matrix has inf/nan values!");
        }

        if (!isMatrixAffine(newMatrix))
        {
            logWarning("Transform matrix is not affine. Setting last row to (0,0,0,1).");
            newMatrix[0][3] = newMatrix[1][3] = newMatrix[2][3] = 0.f;
            newMatrix[3][3] = 1.f;
        }

        return newMatrix;
    }
}
