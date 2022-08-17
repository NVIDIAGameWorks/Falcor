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
#include "USDHelpers.h"
#include "Utils/Logger.h"
#include "Utils/Math/Vector.h"
#include "Scene/SceneBuilder.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/usd/object.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdSkel/tokens.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
END_DISABLE_USD_WARNINGS

#include <string>
#include <functional>

using namespace pxr;

namespace Falcor
{
    class UsdObjHash
    {
    public:
        size_t operator()(const UsdObject& obj) const
        {
            return hash_value(obj);
        }
    };

    struct Float3Hash
    {
        size_t operator()(const float3& v) const
        {
            // Simple hash function that multiplies the integer interpretation of each component by a prime and xors the results
            return (*reinterpret_cast<const uint32_t*>(&v.x) * 7727ULL) ^ (*reinterpret_cast<const uint32_t*>(&v.y) * 5521ULL) ^ (*reinterpret_cast<const uint32_t*>(&v.z) * 6971ULL);
        }
    };

    inline float3 toGlm(const GfVec3f& v)
    {
        return float3(v[0], v[1], v[2]);
    }

    inline float3 toGlm(const GfVec3d& v)
    {
        return float3(v[0], v[1], v[2]);
    }

    inline float3 toGlm(const GfVec3h& v)
    {
        return float3(v[0], v[1], v[2]);
    }

    inline rmcv::mat4 toRMCV(const GfMatrix4d& m)
    {
        // USD uses row-major matrices and row vectors, which are pre-multiplied (v * M) with a matrix to perform a transformation.
        // Falcor RMCV uses row-major matrices and column vectors, which are post-multiplied (M * v).
        // As such, we transpose USD matrices upon import.
        // (Same as for GLM, except we have to write the values actually transposed, rather than using the col-major layout of the constructor)
        return rmcv::mat4({
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3]
            });
    }

    inline SceneBuilder::Node makeNode(const std::string& name, NodeID parentId = NodeID::Invalid())
    {
        return SceneBuilder::Node{name, rmcv::mat4(1.f), rmcv::mat4(1.f), rmcv::mat4(1.f), parentId};
    }

    inline SceneBuilder::Node makeNode(const std::string& name, const rmcv::mat4& xform, const rmcv::mat4& bindTransform, NodeID parentId = NodeID::Invalid())
    {
        return SceneBuilder::Node{name, xform, bindTransform, rmcv::mat4(1.f), parentId};
    }

    inline bool getLocalTransform(const UsdGeomXformable& xformable, rmcv::mat4& xform)
    {
        bool resets = false;
        GfMatrix4d transform;
        xformable.GetLocalTransformation(&transform, &resets, UsdTimeCode::EarliestTime());
        xform = toRMCV(transform);

        return resets;
    }

    // Helper function to return an attribute value, if defined, or the specified default value if not
    template <class T>
    inline T getAttribute(const UsdAttribute& attrib, const T& def)
    {
        T val = def;
        if (attrib)
        {
            attrib.Get(&val, UsdTimeCode::EarliestTime());
        }
        return val;
    }

    using AttributeFrequency = SceneBuilder::Mesh::AttributeFrequency;

    inline size_t computeElementCount(AttributeFrequency freq, size_t faceCount, size_t vertexCount)
    {
        if (freq == AttributeFrequency::Constant)
        {
            return 1;
        }
        else if (freq == AttributeFrequency::Uniform)
        {
            return faceCount;
        }
        else if (freq == AttributeFrequency::Vertex)
        {
            return vertexCount;
        }
        else if (freq == AttributeFrequency::FaceVarying)
        {
            return 3 * faceCount;
        }
        else
        {
            logError("Unsupported primvar interpolation mode {}.", (uint32_t)freq);
            return 0;
        }
    }

    // Compute the count of per-face elements, based on interpolation type
    inline size_t computePerFaceElementCount(AttributeFrequency freq, size_t faceCount)
    {
        if (freq == AttributeFrequency::Uniform)
        {
            return faceCount;
        }
        else if (freq == AttributeFrequency::FaceVarying)
        {
            return 3 * faceCount;
        }
        // Everything else is indexed (vertex, varying), or constant
        return 0;
    }

    inline AttributeFrequency convertInterpolation(const TfToken& mode)
    {
        if (mode == UsdGeomTokens->constant)
        {
            return AttributeFrequency::Constant;
        }
        else if (mode == UsdGeomTokens->uniform)
        {
            return AttributeFrequency::Uniform;
        }
        else if (mode == UsdGeomTokens->vertex || mode == UsdGeomTokens->varying)
        {
            // For our purposes, vertex and varying are synonymous.
            return AttributeFrequency::Vertex;
        }
        else if (mode == UsdGeomTokens->faceVarying)
        {
            return AttributeFrequency::FaceVarying;
        }
        else
        {
            logError("Unknown vertex interpolation mode '{}'.", mode.GetString());
            return AttributeFrequency::None;
        }
    }

    inline TfToken getPurpose(const UsdGeomImageable & prim)
    {
        TfToken purpose = UsdGeomTokens->default_;
        UsdAttribute purposeAttr = prim.GetPurposeAttr();
        if (purposeAttr)
        {
            purposeAttr.Get(&purpose, UsdTimeCode::EarliestTime());
        }
        return purpose;
    }

    inline bool isRenderable(const UsdGeomImageable& geomImageable)
    {
        TfToken purpose = getPurpose(geomImageable);
        if (purpose == UsdGeomTokens->guide || purpose == UsdGeomTokens->proxy)
        {
            return false;
        }
        UsdAttribute visibilityAttr(geomImageable.GetVisibilityAttr());
        if (visibilityAttr)
        {
            TfToken visibility;
            visibilityAttr.Get(&visibility, UsdTimeCode::EarliestTime());
            if (visibility == UsdGeomTokens->invisible)
            {
                return false;
            }
        }

        // Determine the inherited visibility.
        // USD documentation warns that ComputeVisiblity() can be inefficient, as it walks up
        // towards the root every it is called, rather than caching inherited visibility.
        // However, there has yet been no indication that this is a meaningful performance issue in practice.
        return geomImageable.ComputeVisibility() != UsdGeomTokens->invisible;
    }

    inline bool isTimeSampled(const UsdGeomPointBased& geomPointBased)
    {
        return geomPointBased.GetPointsAttr().GetNumTimeSamples() > 1;
    }

    // Route USD messages through Falcor's logging system
    class DiagDelegate : public TfDiagnosticMgr::Delegate
    {
    public:

        void IssueFatalError(const TfCallContext& context, const std::string& msg) override
        {
            reportFatalError(msg + " " + context.GetPrettyFunction());
        }

        void IssueError(const TfError& err) override
        {
            logError(formatMessage(&err));
        }

        void IssueWarning(const TfWarning& warning) override
        {
            logWarning(formatMessage(&warning));
        }

        void IssueStatus(const TfStatus& status) override
        {
            logInfo(formatMessage(&status));
        }

    private:
        std::string formatMessage(const TfDiagnosticBase* elt)
        {
            return elt->GetCommentary();
        }
    };

    class ScopeGuard
    {
    public:
        ScopeGuard(const std::function<void(void)>& func)
            : m_func(func)
        {
        }

        ~ScopeGuard()
        {
            m_func();
        }
    private:
        std::function<void(void)> m_func;
    };
}
