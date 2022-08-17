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
#include "RGLFile.h"
#include "Core/Errors.h"

namespace Falcor
{
    size_t RGLFile::fieldSize(FieldType type)
    {
        switch (type)
        {
        case UInt8: return 1;
        case UInt32: return 4;
        case Float32: return 4;
        default: return 0;
        }
    }

    const RGLFile::Field* RGLFile::getField(const std::string& name) const
    {
        auto iter = mFieldMap.find(name);
        return iter == mFieldMap.end() ? nullptr : &mFields[iter->second];
    }

    void RGLFile::addField(const std::string& name, FieldType type, const std::vector<uint32_t>& shape, const void* data)
    {
        Field field;
        field.name = name;
        field.type = type;
        field.dim = uint32_t(shape.size());
        field.shape.reset(new uint64_t[field.dim]);
        field.numElems = 1;
        for (uint32_t i = 0; i < field.dim; ++i)
        {
            field.shape[i] = shape[i];
            field.numElems *= shape[i];
        }
        size_t N = fieldSize(type);
        if (N == 0) throw RuntimeError("RGLFile::fieldSize: Invalid field type");

        field.data.reset(new uint8_t[N * field.numElems]);
        std::memcpy(field.data.get(), data, N * field.numElems);

        mFieldMap[name] = int(mFields.size());
        mFields.emplace_back(std::move(field));
    }

    void RGLFile::validate()
    {
        const Field* thetaI      = getField("theta_i");
        const Field* phiI        = getField("phi_i");
        const Field* sigma       = getField("sigma");
        const Field* ndf         = getField("ndf");
        const Field* vndf        = getField("vndf");
        const Field* rgb         = getField("rgb");
        const Field* luminance   = getField("luminance");
        const Field* description = getField("description");

        if (!thetaI || thetaI->type != Float32 || thetaI->dim != 1)
        {
            throw RuntimeError("theta_i field missing or invalid");
        }

        if (!phiI || phiI->type != Float32 || phiI->dim != 1)
        {
            throw RuntimeError("phi_i field missing or invalid");
        }

        if (!sigma || sigma->type != Float32 || sigma->dim != 2)
        {
            throw RuntimeError("sigma field missing or invalid");
        }

        if (!ndf || ndf->type != Float32 || ndf->dim != 2)
        {
            throw RuntimeError("ndf field missing or invalid");
        }

        if (!vndf || vndf->type != Float32 || vndf->dim != 4 || vndf->shape[0] != phiI->shape[0]
            || vndf->shape[1] != thetaI->shape[0])
        {
            throw RuntimeError("vndf field missing or invalid");
        }

        if (!luminance || luminance->type != Float32 || luminance->dim != 4
            || luminance->shape[0] != phiI->shape[0]
            || luminance->shape[1] != thetaI->shape[0]
            || luminance->shape[2] != luminance->shape[3])
        {
            throw RuntimeError("luminance field missing or invalid");
        }

        if (!rgb || rgb->type != Float32 || rgb->dim != 5 || rgb->shape[0] != phiI->shape[0]
            || rgb->shape[1] != thetaI->shape[0] || rgb->shape[2] != 3
            || rgb->shape[3] != luminance->shape[2] || rgb->shape[4] != luminance->shape[3])
        {
            throw RuntimeError("rgb field missing or invalid");
        }

        if (!description || description->type != UInt8)
        {
            throw RuntimeError("Description field missing or invalid");
        }

        bool isotropic = phiI->shape[0] <= 2;
        std::string descString(reinterpret_cast<const char*>(description->data.get()), description->numElems);

        mMeasurement = MeasurementData{thetaI, phiI, sigma, ndf, vndf, rgb, luminance, isotropic, std::move(descString)};
    }

    RGLFile::RGLFile(std::ifstream& in)
    {
        auto readBytes = [&](void* dst, size_t size)
        {
            in.read(reinterpret_cast<char*>(dst), size);
        };

        uint8_t header[12];
        readBytes(header, 12);

        uint8_t version[2];
        readBytes(version, 2);

        uint32_t fieldCount;
        readBytes(&fieldCount, 4);

        if (strcmp(reinterpret_cast<const char*>(header), "tensor_file"))
        {
            throw RuntimeError("Invalid file header");
        }
        if (version[0] != 1 || version[1] != 0)
        {
            throw RuntimeError("Unsupported file version");
        }

        for (uint32_t i = 0; i < fieldCount; ++i)
        {
            uint16_t nameLength;
            readBytes(&nameLength, 2);

            std::string fieldName(nameLength, '\0');
            readBytes(&fieldName[0], nameLength);

            uint16_t fieldDim;
            readBytes(&fieldDim, 2);

            uint8_t fieldType;
            readBytes(&fieldType, 1);

            uint64_t offset;
            readBytes(&offset, 8);

            Field field;
            field.name = fieldName;
            field.type = FieldType(fieldType);
            field.dim = fieldDim;
            field.shape.reset(new uint64_t[fieldDim]);
            readBytes(field.shape.get(), 8 * fieldDim);

            if (!in.good()) throw RuntimeError("Error parsing RGL field: File truncated");

            size_t elemSize = fieldSize(FieldType(fieldType));
            if (elemSize == 0)
            {
                // Unsupported field type - ignore, we don't need it.
                continue;
            }

            uint64_t N = 1;
            for (uint32_t j = 0; j < fieldDim; ++j)
            {
                N *= field.shape[j];
            }
            field.numElems = N;

            field.data.reset(new uint8_t[N * elemSize]);

            uint64_t pos = in.tellg();
            in.seekg(offset);
            readBytes(field.data.get(), N * elemSize);
            in.seekg(pos);

            mFieldMap.insert(std::make_pair(std::string(fieldName), int(mFields.size())));
            mFields.emplace_back(std::move(field));
        }

        validate();
    }

    template<typename T>
    void write(std::ofstream& out, const T& t)
    {
        out.write(reinterpret_cast<const char*>(&t), sizeof(T));
    }

    template<typename T>
    void write(std::ofstream& out, const T* t, size_t N)
    {
        out.write(reinterpret_cast<const char*>(t), sizeof(T) * N);
    }

    void RGLFile::saveFile(std::ofstream& out) const
    {
        out.write("tensor_file", 12);

        // Write header.
        uint8_t version[] = {1, 0};
        write(out, version, 2);
        write(out, uint32_t(mFields.size()));

        // Compute size of header+field descriptions.
        uint64_t writeOffset = 18;
        for (const auto& field : mFields)
        {
            writeOffset += 13 + field.name.size() + 8 * field.dim;
        }

        // RGL data blocks seem to be aligned at 8 byte boundaries.
        auto alignAddress = [&](uint64_t a) { return ((a + 7) / 8) * 8; };
        uint64_t alignedOffset = alignAddress(writeOffset);
        uint64_t headerPadding = alignedOffset - writeOffset;

        // Write fields and keep track of where data blocks will start via alignedOffset.
        for (const auto& field : mFields)
        {
            write(out, uint16_t(field.name.size()));
            write(out, field.name.data(), field.name.size());
            write(out, uint16_t(field.dim));
            write(out, uint8_t(field.type));
            write(out, alignedOffset);
            write(out, field.shape.get(), field.dim);
            alignedOffset += fieldSize(field.type) * field.numElems;
            alignedOffset = alignAddress(alignedOffset);
        }
        // Pad with zeros to get correct alignment.
        uint8_t zeros[8] = {0};
        write(out, zeros, headerPadding);
        for (size_t i = 0; i < mFields.size(); ++i)
        {
            size_t length = mFields[i].numElems * fieldSize(mFields[i].type);
            write(out, mFields[i].data.get(), length);
            write(out, zeros, alignAddress(length) - length);
        }
    }
}
