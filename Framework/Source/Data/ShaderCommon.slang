__exported import ShaderCommonImpl;

import Interfaces;
import Lights;
import Material;

type_param TMaterial : IMaterial;

type_param TLightCollection : ILight;

typedef ShadingPoint<TMaterial.BxDF> ShadingData;

ParameterBlock<TMaterial> gMaterial;

typedef LightEnv<TLightCollection> TLightEnvironment;

ParameterBlock<TLightEnvironment> gLightEnv;