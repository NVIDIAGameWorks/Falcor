add_plugin(RadianceCache)

target_sources(RadianceCache PRIVATE
    RadianceCache.cpp
    RadianceCache.h
    MapIslandToTexture.cs.slang
    ClusterTriangles.cs.slang
)

target_source_group(RadianceCache "RenderPasses")
