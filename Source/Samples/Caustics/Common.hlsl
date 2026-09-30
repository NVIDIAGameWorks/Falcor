
#define GATHER_TILE_SIZE_X 32
#define GATHER_TILE_SIZE_Y 32

struct Photon
{
    float3 posW;
    //float3 normalW;
    float3 color;
    float3 dPdx;
    float3 dPdy;
};

struct RayTask
{
    float2 screenCoord;
    float2 pixelSize;
};

struct PixelInfo
{
    uint screenArea;
    uint screenAreaSq;
    uint count;
    int photonIdx;
};

struct RayArgument
{
    int rayTaskCount;
};

struct IDBlock
{
    int address;
    int count;
};

//struct DrawArguments
//{
//    int VertexCountPerInstance;
//    int InstanceCount;
//    int StartVertexLocation;
//    int StartInstanceLocation;
//};

struct DrawArguments
{
    uint IndexCountPerInstance;
    uint InstanceCount;
    uint StartIndexLocation;
    int BaseVertexLocation;
    uint StartInstanceLocation;
};

// на всякий, но вообще я юзаю VSOut из Scene.Raster
struct VertexOut
{
    float2 texC;
    float3 normalW;
    float3 bitangentW;
    float3 posW;
    float3 colorV;
    float3 prevPosH;
    //float3 lightmapC = 0;
};

void getRefractVector(float3 I, float3 N, out float3 R, float eta)
{
    float IN = dot(I, N);
    float RN = -sqrt(1 - eta * eta * (1 - IN * IN));
    float mu = eta * IN - RN;
    R = eta * I - mu * N;
}

// eta = eta i / eta o
bool isTotalInternalReflection(float3 I, float3 N, float eta)
{
    // detect total internal reflection
    float cosI = dot(I, N);
    return cosI * cosI < (1 - 1 / (eta * eta));
}

bool isPhotonAdjecent(Photon photon0, Photon photon1, float normalThreshold, float distanceThreshold, float planarThreshold)
{
    return
        //dot(photon0.normalW, photon1.normalW) < normalThreshold &&
        length(photon0.posW - photon1.posW) < distanceThreshold // &&
        //dot(photon0.normalW, photon0.posW - photon1.posW) < planarThreshold
    ;
}

float smoothKernel(float x)
{
    x = saturate(x);
    return x * x * (2 * x - 3) + 1;
}

bool isInFrustum(float4 p)
{
    return (p.x >= -p.w) && (p.x <= p.w) && (p.y >= -p.w) && (p.y <= p.w) && (p.z >= 0) && (p.z <= p.w);
}

float linearSample(float a, float b, float u)
{
    if (a == b)
        return u;
    float s = 0.5 * (a + b);
    a /= s;
    b /= s;
    return (a - sqrt(lerp(a * a, b * b, u))) / (a - b);
}

float2 bilinearSample(float a00, float a10, float a01, float a11, float2 uv)
{
    float a0 = a00 + a01;
    float a1 = a10 + a11;
    float u = linearSample(a0, a1, uv.x);
    float b0 = lerp(a00, a10, u);
    float b1 = lerp(a01, a11, u);
    float v = linearSample(b0, b1, uv.y);
    return float2(u, v);
}

float bilinearIntepolation(float a00, float a10, float a01, float a11, float2 uv)
{
    float b0 = lerp(a00, a10, uv.x);
    float b1 = lerp(a01, a11, uv.x);
    return lerp(b0, b1, uv.y);
}

float toViewSpace(float4x4 invProj, float depth)
{
    return (invProj[2][2] * depth + invProj[3][2]) / (invProj[2][3] * depth + invProj[3][3]);
}

int getMipOffset(int mip)
{
    return ((1 << (mip << 1)) - 1) / 3;
}

int getMipSize(int mip)
{
    return (1 << mip);
}

int getTextureOffset(int2 pos, int mip)
{
    int mipSize = (1 << mip);
    int mipOffset = ((1 << (mip << 1)) - 1) / 3;
    return mipOffset + pos.y * mipSize + pos.x;
}

int getMipOffset4(int mip)
{
    return (((1 << (mip << 2)) - 1) << 2) / 15;
}

int getMipSize4(int mip)
{
    return ((1 << (mip << 1)) << 1);
}

int getTextureOffset4(int2 pos, int mip)
{
    int mipSize = getMipSize4(mip);
    int mipOffset = getMipOffset4(mip);
    return mipOffset + pos.y * mipSize + pos.x;
}

float getLuminance(float3 color)
{
    return dot(color, float3(0.299f, 0.587f, 0.114f));
}

uint compressColor(float3 color, float scale)
{
    uint3 c = color * 255 * scale;
    return (1 << 28) | (c.r << 19) | (c.g << 9) | (c.b);
}

float4 decompressColor(uint c, float scale)
{
    float4 color;
    color.b = (c & 0x1ff);
    color.g = ((c >> 9) & 0x3ff);
    color.r = ((c >> 19) & 0x1ff);
    color.a = ((c >> 28) & 0xf);
    color.rgb /= (255 * scale);
    return color;
}
