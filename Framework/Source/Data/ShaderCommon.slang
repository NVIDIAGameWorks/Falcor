__exported import ShaderCommonImpl;

import Interfaces;
import Lights;
import Material;

type_param TMaterial : IMaterial;

//type_param TLightEnv : ILight;

typedef ShadingPoint<TMaterial.BxDF> ShadingData;

ParameterBlock<TMaterial> gMaterial;

//ParameterBlock<TLightEnv> gLightEnv;