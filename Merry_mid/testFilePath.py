import os

ASSETS_PATH = "level_Universal"

sound_packs = {
        "percussion": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Percussion/1_Cajon_Tremolo.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Percussion/2_HandclapL_1.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Percussion/3_HandclapR_1.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Percussion/4_Cajon_Side.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Percussion/5_Cajon_Hit.wav",
        },
        "daily": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Quotidien/1_ScrewsInGlass.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Quotidien/2_WaterBottleL.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Quotidien/3_WaterBottleR.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Quotidien/4_PropaneTank.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Quotidien/5_WaterBasin.mp3",
        },
        "animal": {
            "1_Cajon_Tremolo.wav": r"Sound effects/Animal/1_BirdsE.wav",
            "2_HandclapL_1.wav":   r"Sound effects/Animal/2_BirdsB_L.wav",
            "3_HandclapR_1.wav":   r"Sound effects/Animal/3_BirdsB_R.wav",
            "4_Cajon_Side.wav":    r"Sound effects/Animal/4_BirdsD.wav",
            "5_Cajon_Hit.wav":     r"Sound effects/Animal/5_BirdsA.wav",
        }
    }
 
full_path = os.path.join(ASSETS_PATH, sound_packs.get('percussion').get('1_Cajon_Tremolo.wav')) # 組合完整路徑
if os.path.exists(sound_packs.get('percussion').get('1_Cajon_Tremolo.wav')):
    print(123)