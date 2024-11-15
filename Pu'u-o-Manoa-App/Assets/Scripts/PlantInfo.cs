using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class PlantInfo
{
    public string Genus;
    public string Species;
    public List<string> CommonNames;
    public List<string> Synonyms;
    public string DidYouKnow;
    public string DistributionStatus;
    public string EndangeredSpeciesStatus;
    public List<string> PlantFormGrowthHabit;
    public List<string> MatureSizeHeight;
    public string LifeSpan;
    public List<string> LandscapeUses;
    public string AdditionalLandscapeUseInfo;
    public List<string> SourceOfFragrance;
    public string PlantProducesFlowers;
    public string FlowerType;
    public List<string> FlowerColors;
    public string AdditionalFlowerColorInfo;
    public List<string> BloomingPeriod;
    public string AdditionalBloomingPeriodInfo;
    public List<string> PlantTexture;
    public List<string> LeafColors;
    public string AdditionalLeafColorInfo;
    public string AdditionalPestDiseaseInfo;
    public string PruningInfo;
    public List<string> WaterRequirements;
    public string AdditionalWaterInfo;
    public bool SoilMustBeWellDrained;
    public List<string> LightConditions;
    public string AdditionalLightingInfo;
    public List<string> Tolerances;
    public List<string> Soils;
    public List<string> NaturalRange;
    public List<string> NaturalZones;
    public List<string> Habitat;
    public string AdditionalHabitatInfo;
    public string GeneralInfo;
    public string Etymology;
    public string BackgroundInfo;
    public string ModernUse;
    public string AdditionalReferences;
}

[Serializable]
public class PlantData
{
    public Dictionary<string, PlantInfo> Plants = new Dictionary<string, PlantInfo>(); // This will hold the plant information
}