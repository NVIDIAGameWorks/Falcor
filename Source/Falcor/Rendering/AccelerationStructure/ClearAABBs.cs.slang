import Utils.Math.AABB;

cbuffer CB {
    uint gMax; //Max number of elements
    uint gOffset; // Index offset
    uint gCounterIdx; //Index for the counter
    bool gClearToNaN; // If true minPoint.x is cleared to nan
}

RWStructuredBuffer<AABB> gAABB;
RWStructuredBuffer<uint> gCounter;

static const bool kUseCounter = USE_COUNTER_TO_CLEAR;

[numthreads(256, 1, 1)]
void main(uint thdID: SV_DispatchThreadID)
{
    const uint aabbIdx = thdID + gOffset;
    if (kUseCounter) {
        const uint counter = gCounter[gCounterIdx];
        if (thdID < counter)
            return;
    }
    if (aabbIdx >= gMax)
        return;
    AABB aabb = {};
    aabb.invalidate();
    if(gClearToNaN)
        aabb.minPoint.x = 0.0 / 0.0;

    gAABB[aabbIdx] = aabb;
}
