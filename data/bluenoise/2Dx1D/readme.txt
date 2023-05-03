This is a spatio-temporal blue noise mask, or 2Dx1D blue noise.
64x64x64
made with the void and cluster algorithm modified such that pixels only contribute energy to each other if they are from the same z slice or the same (x,y) location but a different z.
Each of these textures are fully blue noise, but also, each pixel individually is also 1D blue noise over the z axis.

There is also 128x128x64 blue noise in here.