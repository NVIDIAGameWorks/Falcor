The retargetting feature of the Heitz/Belcour technique needs to know how to move seeds from this frame to the next to get better results.

My understanding of it, is that this lets seeds move beyond tile boundaries to get a more globally optimal result.

This texture gives that info.

The texture was made by randomly swapping pixels to reduce RMSE between bn_2D_0 and an R2 shifted version of that texture.

No pixel should move more than 6 units away, to keep seeds from doing large moves which could cause them to render significantly differently.

subtract 6 from the uint2(RG*255.0) value read from the texture to get the actual offset amounts