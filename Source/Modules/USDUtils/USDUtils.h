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
#include "USDHelpers.h"
#include "Core/Error.h"
#include "Utils/Logger.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
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
    size_t operator()(const UsdObject& obj) const { return hash_value(obj); }
};

struct Float3Hash
{
    size_t operator()(const float3& v) const
    {
        // Simple hash function that multiplies the integer interpretation of each component by a prime and xors the results
        return (*reinterpret_cast<const uint32_t*>(&v.x) * 7727ULL) ^ (*reinterpret_cast<const uint32_t*>(&v.y) * 5521ULL) ^
               (*reinterpret_cast<const uint32_t*>(&v.z) * 6971ULL);
    }
};

inline float3 toFalcor(const GfVec3f& v)
{
    return float3(v[0], v[1], v[2]);
}

inline float3 toFalcor(const GfVec3d& v)
{
    return float3(v[0], v[1], v[2]);
}

inline float3 toFalcor(const GfVec3h& v)
{
    return float3(v[0], v[1], v[2]);
}

inline float4x4 toFalcor(const GfMatrix4d& m)
{
    // USD uses row-major matrices and row vectors, which are pre-multiplied (v * M) with a matrix to perform a transformation.
    // Falcor uses row-major matrices and column vectors, which are post-multiplied (M * v).
    // As such, we transpose USD matrices upon import.
    return float4x4({
        // clang-format off
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3]
        // clang-format on
    });
}

inline float4x4 toFalcor(const GfMatrix4f& m)
{
    // USD uses row-major matrices and row vectors, which are pre-multiplied (v * M) with a matrix to perform a transformation.
    // Falcor uses row-major matrices and column vectors, which are post-multiplied (M * v).
    // As such, we transpose USD matrices upon import.
    return float4x4({
        // clang-format off
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3]
        // clang-format on
    });
}

inline bool getLocalTransform(const UsdGeomXformable& xformable, float4x4& xform)
{
    bool resets = false;
    GfMatrix4d transform;
    xformable.GetLocalTransformation(&transform, &resets, UsdTimeCode::EarliestTime());
    xform = toFalcor(transform);

    return resets;
}

// Helper function to return an attribute value, if defined, or the specified default value if not
template<class T>
inline T getAttribute(const UsdAttribute& attrib, const T& def)
{
    T val = def;
    if (attrib)
    {
        attrib.Get(&val, UsdTimeCode::EarliestTime());
    }
    return val;
}

// Helper function that returns the main attribute if it is authored (non-default),
// otherwise it tries to retrieve the fallback attribute and should that fail, the default value.
// This is used to provide compatibility, e.g. when the default disk light radius changed from "radius" to "inputs:radius"
template<class T>
inline T getAuthoredAttribute(const UsdAttribute& mainAttrib, const UsdAttribute& fallbackAttrib, const T& def)
{
    T val = def;
    if (mainAttrib && mainAttrib.IsAuthored())
    {
        mainAttrib.Get(&val, UsdTimeCode::EarliestTime());
    }
    else if (fallbackAttrib)
    {
        fallbackAttrib.Get(&val, UsdTimeCode::EarliestTime());
    }
    return val;
}

template<class T>
inline T getAttribute(const UsdPrim& prim, const std::string& attribName, const T& def)
{
    T val = def;
    UsdAttribute attrib = prim.GetAttribute(TfToken(attribName));
    if (attrib.IsValid())
    {
        attrib.Get(&val, UsdTimeCode::EarliestTime());
    }
    return val;
}

inline TfToken getPurpose(const UsdGeomImageable& prim)
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
        FALCOR_THROW(msg + " " + context.GetPrettyFunction());
    }

    void IssueError(const TfError& err) override { logError(formatMessage(&err)); }

    void IssueWarning(const TfWarning& warning) override { logWarning(formatMessage(&warning)); }

    void IssueStatus(const TfStatus& status) override { logInfo(formatMessage(&status)); }

private:
    std::string formatMessage(const TfDiagnosticBase* elt) { return elt->GetCommentary(); }
};

class ScopeGuard
{
public:
    ScopeGuard(const std::function<void(void)>& func) : m_func(func) {}

    ~ScopeGuard() { m_func(); }

private:
    std::function<void(void)> m_func;
};
} // namespace Falcor
