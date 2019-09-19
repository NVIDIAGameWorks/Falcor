
RWStructuredBuffer<float> result;

cbuffer TestCB
{
    int nValues;
    float scale;
};

void main()
{
    for (int i = 0; i < nValues; ++i)
    {
        result[i] = scale * i;
    }
}
