struct type_7_ {
    member: array<vec4<f32>>,
}

struct type_16_ {
    member: u32,
    member_1_: u32,
}

struct type_17_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: f32,
    member_6_: f32,
    member_7_: u32,
}

struct type_18_ {
    member: type_17_,
}

struct type_24_ {
    member: array<f32, 4>,
}

struct type_25_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_27_ {
    member: type_24_,
    member_1_: type_24_,
}

struct type_32_ {
    member: type_27_,
    member_1_: type_27_,
    member_2_: type_27_,
    member_3_: type_27_,
}

struct type_33_ {
    member: type_24_,
    member_1_: type_24_,
    member_2_: type_24_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> num_wg_1_: vec3<u32>;
@group(0) @binding(0) 
var<uniform> uniforms: type_18_;
@group(0) @binding(1) 
var<storage> core_inputs: type_7_;
@group(0) @binding(2) 
var<storage> quad_params: type_7_;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(3) 
var<storage, read_write> tile_checkpoints: type_7_;
@group(0) @binding(4) 
var<storage, read_write> feature_outputs: type_7_;
var<workgroup> lane_buf: array<vec4<f32>, 512>;
var<workgroup> aux_scratch: array<f32, 256>;

fn function_() {
    var local: array<f32, 4>;
    var local_1_: array<f32, 4>;
    var local_2_: array<f32, 4>;
    var local_3_: array<f32, 4>;
    var local_4_: array<f32, 4>;
    var local_5_: array<f32, 4>;
    var local_6_: array<f32, 4>;
    var local_7_: array<f32, 4>;
    var local_8_: array<f32, 4>;
    var local_9_: array<f32, 4>;
    var local_10_: array<f32, 4>;
    var local_11_: array<f32, 4>;
    var phi_227_: u32;
    var phi_258_: u32;
    var phi_269_: u32;
    var phi_272_: u32;
    var phi_270_: u32;
    var phi_273_: u32;
    var phi_284_: u32;
    var phi_287_: u32;
    var phi_289_: u32;
    var phi_291_: u32;
    var phi_305_: u32;
    var phi_306_: u32;
    var phi_285_: u32;
    var phi_288_: u32;
    var phi_290_: u32;
    var phi_292_: u32;
    var local_12_: u32;
    var phi_310_: u32;
    var phi_311_: u32;
    var phi_325_: u32;
    var phi_326_: u32;
    var phi_340_: vec4<f32>;
    var phi_353_: vec4<f32>;
    var phi_366_: vec4<f32>;
    var phi_379_: vec4<f32>;
    var phi_392_: vec4<f32>;
    var phi_405_: vec4<f32>;
    var phi_418_: vec4<f32>;
    var phi_431_: vec4<f32>;
    var phi_444_: vec4<f32>;
    var phi_457_: vec4<f32>;
    var phi_470_: vec4<f32>;
    var phi_483_: vec4<f32>;
    var phi_496_: vec4<f32>;
    var phi_509_: vec4<f32>;
    var phi_522_: vec4<f32>;
    var phi_535_: vec4<f32>;
    var phi_548_: vec4<f32>;
    var phi_561_: vec4<f32>;
    var phi_574_: vec4<f32>;
    var phi_587_: vec4<f32>;
    var phi_600_: vec4<f32>;
    var phi_613_: vec4<f32>;
    var phi_626_: vec4<f32>;
    var phi_666_: bool;
    var phi_672_: f32;
    var phi_673_: f32;
    var phi_680_: f32;
    var phi_706_: bool;
    var phi_712_: f32;
    var phi_713_: f32;
    var phi_720_: f32;
    var phi_746_: bool;
    var phi_752_: f32;
    var phi_753_: f32;
    var phi_760_: f32;
    var phi_786_: bool;
    var phi_792_: f32;
    var phi_793_: f32;
    var phi_800_: f32;
    var phi_832_: vec4<f32>;
    var phi_844_: bool;
    var phi_861_: bool;
    var phi_878_: bool;
    var phi_895_: bool;
    var phi_914_: vec4<f32>;
    var phi_926_: bool;
    var phi_943_: bool;
    var phi_960_: bool;
    var phi_977_: bool;
    var phi_996_: vec4<f32>;
    var phi_1008_: bool;
    var phi_1025_: bool;
    var phi_1042_: bool;
    var phi_1059_: bool;
    var phi_1078_: vec4<f32>;
    var phi_1090_: bool;
    var phi_1107_: bool;
    var phi_1124_: bool;
    var phi_1141_: bool;
    var phi_1160_: vec4<f32>;
    var phi_1172_: bool;
    var phi_1189_: bool;
    var phi_1206_: bool;
    var phi_1223_: bool;
    var phi_1242_: vec4<f32>;
    var phi_1254_: bool;
    var phi_1271_: bool;
    var phi_1288_: bool;
    var phi_1305_: bool;
    var phi_1324_: vec4<f32>;
    var phi_1336_: bool;
    var phi_1353_: f32;
    var phi_1365_: bool;
    var phi_1382_: f32;
    var phi_1394_: bool;
    var phi_1411_: f32;
    var phi_1423_: bool;
    var phi_1440_: f32;
    var phi_1454_: vec4<f32>;
    var phi_1466_: bool;
    var phi_1483_: f32;
    var phi_1495_: bool;
    var phi_1512_: f32;
    var phi_1524_: bool;
    var phi_1541_: f32;
    var phi_1553_: bool;
    var phi_1570_: f32;
    var phi_1581_: type_32_;
    var phi_1583_: type_32_;
    var phi_1584_: bool;
    var phi_1587_: bool;
    var phi_1588_: type_32_;
    var phi_1589_: bool;
    var phi_1591_: bool;
    var phi_1592_: bool;
    var phi_1614_: type_32_;
    var phi_16480_: bool;
    var phi_1618_: type_16_;
    var phi_1621_: type_32_;
    var phi_1619_: type_16_;
    var phi_1644_: type_16_;
    var phi_1658_: u32;
    var phi_1678_: bool;
    var phi_1727_: bool;
    var phi_1743_: bool;
    var phi_1759_: bool;
    var phi_1775_: bool;
    var phi_1807_: bool;
    var phi_1823_: bool;
    var phi_1839_: bool;
    var phi_1855_: bool;
    var phi_1887_: bool;
    var phi_1903_: bool;
    var phi_1919_: bool;
    var phi_1935_: bool;
    var phi_1967_: bool;
    var phi_1983_: bool;
    var phi_1999_: bool;
    var phi_2015_: bool;
    var phi_2047_: bool;
    var phi_2063_: bool;
    var phi_2079_: bool;
    var phi_2095_: bool;
    var phi_2127_: bool;
    var phi_2143_: bool;
    var phi_2159_: bool;
    var phi_2175_: bool;
    var phi_2207_: bool;
    var phi_2223_: bool;
    var phi_2239_: bool;
    var phi_2255_: bool;
    var phi_2287_: bool;
    var phi_2303_: bool;
    var phi_2319_: bool;
    var phi_2335_: bool;
    var phi_2382_: vec4<f32>;
    var phi_2395_: vec4<f32>;
    var phi_2396_: vec4<f32>;
    var phi_2397_: vec4<f32>;
    var phi_16474_: bool;
    var phi_2414_: type_16_;
    var phi_2417_: type_32_;
    var phi_2415_: type_16_;
    var phi_2440_: type_16_;
    var phi_2458_: f32;
    var phi_2469_: f32;
    var phi_2480_: f32;
    var phi_2491_: f32;
    var phi_2502_: f32;
    var phi_2513_: f32;
    var phi_2535_: f32;
    var phi_2546_: f32;
    var phi_2557_: f32;
    var phi_2568_: f32;
    var phi_2579_: f32;
    var phi_2590_: f32;
    var phi_2601_: f32;
    var phi_2612_: f32;
    var phi_2623_: f32;
    var phi_2634_: f32;
    var phi_2659_: bool;
    var phi_2665_: f32;
    var phi_2666_: f32;
    var phi_2673_: f32;
    var phi_2698_: bool;
    var phi_2704_: f32;
    var phi_2705_: f32;
    var phi_2712_: f32;
    var phi_2737_: bool;
    var phi_2743_: f32;
    var phi_2744_: f32;
    var phi_2751_: f32;
    var phi_2776_: bool;
    var phi_2782_: f32;
    var phi_2783_: f32;
    var phi_2790_: f32;
    var phi_2815_: bool;
    var phi_2821_: f32;
    var phi_2822_: f32;
    var phi_2829_: f32;
    var phi_2854_: bool;
    var phi_2860_: f32;
    var phi_2861_: f32;
    var phi_2868_: f32;
    var phi_2939_: f32;
    var phi_2950_: f32;
    var phi_2961_: f32;
    var phi_2972_: f32;
    var phi_2983_: f32;
    var phi_3047_: f32;
    var phi_3058_: bool;
    var phi_3064_: f32;
    var phi_3078_: f32;
    var phi_3089_: bool;
    var phi_3095_: f32;
    var phi_3109_: type_25_;
    var phi_3187_: u32;
    var phi_3190_: type_25_;
    var phi_3286_: type_25_;
    var phi_3188_: u32;
    var phi_3191_: type_25_;
    var phi_16434_: bool;
    var phi_3398_: array<f32, 2>;
    var phi_3410_: u32;
    var phi_3468_: array<f32, 2>;
    var phi_3729_: type_25_;
    var phi_3807_: u32;
    var phi_3810_: type_25_;
    var phi_3906_: type_25_;
    var phi_3808_: u32;
    var phi_3811_: type_25_;
    var phi_16747_: bool;
    var phi_4018_: array<f32, 2>;
    var phi_4030_: u32;
    var phi_4088_: array<f32, 2>;
    var phi_4338_: f32;
    var phi_4431_: type_16_;
    var phi_4434_: f32;
    var phi_4436_: f32;
    var phi_4432_: type_16_;
    var phi_4459_: type_16_;
    var phi_4512_: u32;
    var phi_4513_: u32;
    var phi_11134_: bool;
    var phi_11194_: bool;
    var phi_11254_: bool;
    var phi_11314_: bool;
    var phi_11374_: bool;
    var phi_11434_: bool;
    var phi_11494_: bool;
    var phi_11554_: bool;
    var phi_5254_: u32;
    var phi_5255_: u32;
    var phi_5316_: u32;
    var phi_5317_: u32;
    var phi_11734_: bool;
    var phi_11794_: bool;
    var phi_11854_: bool;
    var phi_11914_: bool;
    var phi_11974_: bool;
    var phi_12034_: bool;
    var phi_12094_: bool;
    var phi_12154_: bool;
    var phi_6058_: u32;
    var phi_6059_: u32;
    var phi_4435_: f32;
    var phi_4437_: f32;
    var phi_16756_: bool;
    var local_13_: f32;
    var local_14_: f32;
    var phi_17048_: bool;
    var phi_6201_: type_32_;
    var phi_6213_: f32;
    var phi_6225_: f32;
    var phi_12334_: bool;
    var phi_12349_: bool;
    var phi_6263_: bool;
    var phi_6269_: f32;
    var phi_6287_: bool;
    var phi_6293_: f32;
    var phi_17037_: bool;
    var phi_2418_: type_32_;
    var phi_17036_: bool;
    var phi_6346_: u32;
    var phi_6347_: u32;
    var phi_6361_: u32;
    var phi_6362_: u32;
    var phi_6376_: u32;
    var phi_6377_: u32;
    var phi_6391_: u32;
    var phi_6392_: u32;
    var phi_6414_: u32;
    var phi_6415_: u32;
    var phi_6429_: u32;
    var phi_6430_: u32;
    var phi_6444_: u32;
    var phi_6445_: u32;
    var phi_6459_: u32;
    var phi_6460_: u32;
    var local_15_: type_32_;
    var phi_6485_: u32;
    var phi_6486_: u32;
    var phi_6500_: u32;
    var phi_6501_: u32;
    var phi_6515_: u32;
    var phi_6516_: u32;
    var phi_6530_: u32;
    var phi_6531_: u32;
    var phi_6553_: u32;
    var phi_6554_: u32;
    var phi_6568_: u32;
    var phi_6569_: u32;
    var phi_6583_: u32;
    var phi_6584_: u32;
    var phi_6598_: u32;
    var phi_6599_: u32;
    var local_16_: type_32_;
    var phi_6670_: type_24_;
    var phi_6671_: type_24_;
    var phi_12364_: bool;
    var phi_12379_: bool;
    var phi_12394_: bool;
    var phi_12409_: bool;
    var phi_12424_: bool;
    var phi_12439_: bool;
    var phi_12454_: bool;
    var phi_12469_: bool;
    var phi_12484_: bool;
    var phi_12499_: bool;
    var phi_12514_: bool;
    var phi_12529_: bool;
    var phi_12544_: bool;
    var phi_12559_: bool;
    var phi_12574_: bool;
    var phi_12589_: bool;
    var phi_12604_: bool;
    var phi_12619_: bool;
    var phi_12634_: bool;
    var phi_12649_: bool;
    var phi_12664_: bool;
    var phi_12679_: bool;
    var phi_12694_: bool;
    var phi_12709_: bool;
    var phi_12724_: bool;
    var phi_12739_: bool;
    var phi_12754_: bool;
    var phi_12769_: bool;
    var phi_12784_: bool;
    var phi_12799_: bool;
    var phi_12814_: bool;
    var phi_12829_: bool;
    var phi_12844_: bool;
    var phi_12859_: bool;
    var phi_12874_: bool;
    var phi_12889_: bool;
    var phi_12904_: bool;
    var phi_12919_: bool;
    var phi_12934_: bool;
    var phi_12949_: bool;
    var phi_12964_: bool;
    var phi_12979_: bool;
    var phi_12994_: bool;
    var phi_13009_: bool;
    var phi_13024_: bool;
    var phi_13039_: bool;
    var phi_13054_: bool;
    var phi_13069_: bool;
    var phi_13084_: bool;
    var phi_13099_: bool;
    var phi_13114_: bool;
    var phi_13129_: bool;
    var phi_13144_: bool;
    var phi_13159_: bool;
    var phi_13174_: bool;
    var phi_13189_: bool;
    var phi_13204_: bool;
    var phi_13219_: bool;
    var phi_13234_: bool;
    var phi_13249_: bool;
    var phi_13264_: bool;
    var phi_13279_: bool;
    var phi_13294_: bool;
    var phi_13309_: bool;
    var phi_7106_: bool;
    var phi_7108_: bool;
    var phi_7109_: bool;
    var phi_7110_: bool;
    var phi_7111_: bool;
    var phi_13324_: bool;
    var phi_13339_: bool;
    var phi_13354_: bool;
    var phi_13369_: bool;
    var phi_7489_: type_33_;
    var phi_13384_: bool;
    var phi_13399_: bool;
    var phi_13414_: bool;
    var phi_13429_: bool;
    var phi_13444_: bool;
    var phi_13459_: bool;
    var phi_13474_: bool;
    var phi_13489_: bool;
    var phi_13504_: bool;
    var phi_13519_: bool;
    var phi_13534_: bool;
    var phi_13549_: bool;
    var phi_13564_: bool;
    var phi_13579_: bool;
    var phi_13594_: bool;
    var phi_13609_: bool;
    var phi_13624_: bool;
    var phi_13639_: bool;
    var phi_13654_: bool;
    var phi_13669_: bool;
    var phi_13684_: bool;
    var phi_13699_: bool;
    var phi_13714_: bool;
    var phi_13729_: bool;
    var phi_13744_: bool;
    var phi_13759_: bool;
    var phi_13774_: bool;
    var phi_13789_: bool;
    var phi_13804_: bool;
    var phi_13819_: bool;
    var phi_13834_: bool;
    var phi_13849_: bool;
    var phi_13864_: bool;
    var phi_13879_: bool;
    var phi_13894_: bool;
    var phi_13909_: bool;
    var phi_13924_: bool;
    var phi_13939_: bool;
    var phi_13954_: bool;
    var phi_13969_: bool;
    var phi_13984_: bool;
    var phi_13999_: bool;
    var phi_14014_: bool;
    var phi_14029_: bool;
    var phi_14044_: bool;
    var phi_14059_: bool;
    var phi_14074_: bool;
    var phi_14089_: bool;
    var phi_14104_: bool;
    var phi_14119_: bool;
    var phi_14134_: bool;
    var phi_14149_: bool;
    var phi_14164_: bool;
    var phi_14179_: bool;
    var phi_14194_: bool;
    var phi_14209_: bool;
    var phi_14224_: bool;
    var phi_14239_: bool;
    var phi_14254_: bool;
    var phi_14269_: bool;
    var phi_14284_: bool;
    var phi_14299_: bool;
    var phi_14314_: bool;
    var phi_14329_: bool;
    var phi_7925_: bool;
    var phi_7927_: bool;
    var phi_7928_: bool;
    var phi_7929_: bool;
    var phi_7930_: bool;
    var phi_14344_: bool;
    var phi_14359_: bool;
    var phi_14374_: bool;
    var phi_14389_: bool;
    var phi_8308_: type_33_;
    var phi_8325_: bool;
    var phi_8327_: bool;
    var phi_8328_: bool;
    var phi_8329_: bool;
    var phi_8330_: bool;
    var phi_14404_: bool;
    var phi_14419_: bool;
    var phi_14434_: bool;
    var phi_14449_: bool;
    var phi_14464_: bool;
    var phi_14479_: bool;
    var phi_14494_: bool;
    var phi_14509_: bool;
    var phi_14524_: bool;
    var phi_14539_: bool;
    var phi_14554_: bool;
    var phi_14569_: bool;
    var phi_14584_: bool;
    var phi_14599_: bool;
    var phi_14614_: bool;
    var phi_14629_: bool;
    var phi_14644_: bool;
    var phi_14659_: bool;
    var phi_14674_: bool;
    var phi_14689_: bool;
    var phi_14704_: bool;
    var phi_14719_: bool;
    var phi_14734_: bool;
    var phi_14749_: bool;
    var phi_14764_: bool;
    var phi_14779_: bool;
    var phi_14794_: bool;
    var phi_14809_: bool;
    var phi_14824_: bool;
    var phi_14839_: bool;
    var phi_14854_: bool;
    var phi_14869_: bool;
    var phi_14884_: bool;
    var phi_14899_: bool;
    var phi_14914_: bool;
    var phi_14929_: bool;
    var phi_14944_: bool;
    var phi_14959_: bool;
    var phi_14974_: bool;
    var phi_14989_: bool;
    var phi_15004_: bool;
    var phi_15019_: bool;
    var phi_15034_: bool;
    var phi_15049_: bool;
    var phi_15304_: bool;
    var phi_15319_: bool;
    var phi_15334_: bool;
    var phi_15349_: bool;
    var phi_15364_: bool;
    var phi_15379_: bool;
    var phi_15394_: bool;
    var phi_15409_: bool;
    var phi_15424_: bool;
    var phi_15439_: bool;
    var phi_15454_: bool;
    var phi_15469_: bool;
    var phi_15484_: bool;
    var phi_15499_: bool;
    var phi_15514_: bool;
    var phi_15529_: bool;
    var phi_15544_: bool;
    var phi_15559_: bool;
    var phi_15574_: bool;
    var phi_15589_: bool;
    var phi_15604_: bool;
    var phi_15619_: bool;
    var phi_15634_: bool;
    var phi_15649_: bool;
    var phi_15664_: bool;
    var phi_15679_: bool;
    var phi_15694_: bool;
    var phi_15709_: bool;
    var phi_15724_: bool;
    var phi_15739_: bool;
    var phi_15754_: bool;
    var phi_15769_: bool;
    var phi_15784_: bool;
    var phi_15799_: bool;
    var phi_15814_: bool;
    var phi_15829_: bool;
    var phi_15844_: bool;
    var phi_15859_: bool;
    var phi_15874_: bool;
    var phi_15889_: bool;
    var phi_15904_: bool;
    var phi_15919_: bool;
    var phi_15934_: bool;
    var phi_15949_: bool;
    var phi_15964_: bool;
    var phi_15979_: bool;
    var phi_15994_: bool;
    var phi_16009_: bool;
    var phi_16024_: bool;
    var phi_16039_: bool;
    var phi_16054_: bool;
    var phi_16069_: bool;
    var phi_16324_: bool;
    var phi_16339_: bool;
    var phi_16354_: bool;
    var phi_16369_: bool;
    var phi_16384_: bool;
    var phi_16399_: bool;
    var phi_16414_: bool;
    var phi_16429_: bool;
    var phi_10218_: type_24_;
    var phi_10219_: type_24_;
    var phi_10380_: bool;
    var phi_10396_: bool;
    var phi_10412_: bool;
    var phi_10428_: bool;
    var phi_10466_: bool;
    var phi_10482_: bool;
    var phi_10498_: bool;
    var phi_10514_: bool;
    var phi_10546_: bool;
    var phi_10562_: bool;
    var phi_10578_: bool;
    var phi_10594_: bool;
    var phi_10626_: bool;
    var phi_10642_: bool;
    var phi_10658_: bool;
    var phi_10674_: bool;
    var phi_10706_: bool;
    var phi_10722_: bool;
    var phi_10738_: bool;
    var phi_10754_: bool;
    var phi_10786_: bool;
    var phi_10802_: bool;
    var phi_10818_: bool;
    var phi_10834_: bool;
    var phi_10875_: bool;
    var phi_10892_: f32;
    var phi_10904_: bool;
    var phi_10921_: f32;
    var phi_10933_: bool;
    var phi_10950_: f32;
    var phi_10962_: bool;
    var phi_10979_: f32;
    var phi_11000_: bool;
    var phi_11017_: f32;
    var phi_11029_: bool;
    var phi_11046_: f32;
    var phi_11058_: bool;
    var phi_11075_: f32;
    var phi_11087_: bool;
    var phi_11104_: f32;
    var phi_11109_: type_32_;
    var phi_17054_: bool;
    var phi_1622_: type_32_;
    var phi_17053_: bool;
    var local_17_: u32;
    var local_18_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e103_ = wg_id_1_;
            let _e104_ = num_wg_1_;
            let _e105_ = lid_1_;
            let _e107_ = arrayLength((&core_inputs.member));
            let _e109_ = arrayLength((&quad_params.member));
            let _e111_ = arrayLength((&tile_checkpoints.member));
            let _e113_ = arrayLength((&feature_outputs.member));
            let _e114_ = (_e105_ & 255u);
            let _e122_ = select(1u, _e104_.y, (1u < _e104_.y));
            let _e125_ = ((_e103_.z * select(1u, _e104_.x, (1u < _e104_.x))) + _e103_.x);
            let _e128_ = uniforms.member.member;
            if (_e128_ > 0u) {
                phi_227_ = (((_e128_ - 1u) >> bitcast<u32>(8i)) + 1u);
            } else {
                phi_227_ = 0u;
            }
            let _e135_ = phi_227_;
            let _e138_ = uniforms.member.member_1_;
            let _e140_ = select(1u, _e138_, (1u < _e138_));
            let _e142_ = select(_e138_, _e135_, (_e135_ < _e138_));
            let _e145_ = uniforms.member.member_4_;
            let _e147_ = select(_e145_, _e142_, (_e142_ < _e145_));
            let _e148_ = (_e122_ <= 1u);
            if _e148_ {
                phi_325_ = _e147_;
                phi_326_ = _e142_;
            } else {
                if (_e142_ > _e147_) {
                    phi_258_ = (_e142_ - _e147_);
                } else {
                    phi_258_ = 0u;
                }
                let _e152_ = phi_258_;
                if (_e152_ == 0u) {
                    phi_311_ = 0u;
                } else {
                    if _e148_ {
                        phi_310_ = _e152_;
                    } else {
                        let _e154_ = (_e152_ - 1u);
                        phi_269_ = 1u;
                        phi_272_ = _e122_;
                        loop {
                            let _e156_ = phi_269_;
                            let _e158_ = phi_272_;
                            local_17_ = _e156_;
                            local_18_ = _e158_;
                            let _e161_ = (_e158_ <= (_e154_ >> bitcast<u32>(1i)));
                            if _e161_ {
                                phi_270_ = (_e156_ << bitcast<u32>(1i));
                                phi_273_ = (_e158_ << bitcast<u32>(1i));
                            } else {
                                phi_270_ = u32();
                                phi_273_ = u32();
                            }
                            let _e167_ = phi_270_;
                            let _e169_ = phi_273_;
                            phi_269_ = _e167_;
                            phi_272_ = _e169_;
                            if !(_e161_) {
                                break;
                            }
                        }
                        phi_284_ = 0u;
                        let _e5607_ = local_17_;
                        phi_287_ = _e5607_;
                        phi_289_ = _e154_;
                        let _e5611_ = local_18_;
                        phi_291_ = _e5611_;
                        loop {
                            let _e172_ = phi_284_;
                            let _e174_ = phi_287_;
                            let _e176_ = phi_289_;
                            let _e178_ = phi_291_;
                            local_12_ = _e172_;
                            let _e179_ = (_e174_ > 0u);
                            if _e179_ {
                                if (_e176_ >= _e178_) {
                                    phi_305_ = (_e172_ | _e174_);
                                    phi_306_ = (_e176_ - _e178_);
                                } else {
                                    phi_305_ = _e172_;
                                    phi_306_ = _e176_;
                                }
                                let _e184_ = phi_305_;
                                let _e186_ = phi_306_;
                                phi_285_ = _e184_;
                                phi_288_ = (_e174_ >> bitcast<u32>(1i));
                                phi_290_ = _e186_;
                                phi_292_ = (_e178_ >> bitcast<u32>(1i));
                            } else {
                                phi_285_ = u32();
                                phi_288_ = u32();
                                phi_290_ = u32();
                                phi_292_ = u32();
                            }
                            let _e192_ = phi_285_;
                            let _e194_ = phi_288_;
                            let _e196_ = phi_290_;
                            let _e198_ = phi_292_;
                            phi_284_ = _e192_;
                            phi_287_ = _e194_;
                            phi_289_ = _e196_;
                            phi_291_ = _e198_;
                            if !(_e179_) {
                                break;
                            }
                        }
                        let _e201_ = local_12_;
                        phi_310_ = (_e201_ + 1u);
                    }
                    let _e204_ = phi_310_;
                    phi_311_ = _e204_;
                }
                let _e206_ = phi_311_;
                let _e208_ = (_e147_ + (_e103_.y * _e206_));
                let _e210_ = select(_e208_, _e142_, (_e142_ < _e208_));
                let _e211_ = (_e210_ + _e206_);
                phi_325_ = _e210_;
                phi_326_ = select(_e211_, _e142_, (_e142_ < _e211_));
            }
            let _e215_ = phi_325_;
            let _e217_ = phi_326_;
            let _e218_ = (_e125_ * 24u);
            let _e219_ = (_e218_ < _e109_);
            if _e219_ {
                if _e219_ {
                } else {
                    break;
                }
                let _e222_ = quad_params.member[_e218_];
                phi_340_ = _e222_;
            } else {
                phi_340_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e224_ = phi_340_;
            let _e225_ = (_e218_ + 1u);
            let _e226_ = (_e225_ < _e109_);
            if _e226_ {
                if _e226_ {
                } else {
                    break;
                }
                let _e229_ = quad_params.member[_e225_];
                phi_353_ = _e229_;
            } else {
                phi_353_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e231_ = phi_353_;
            let _e232_ = (_e218_ + 2u);
            let _e233_ = (_e232_ < _e109_);
            if _e233_ {
                if _e233_ {
                } else {
                    break;
                }
                let _e236_ = quad_params.member[_e232_];
                phi_366_ = _e236_;
            } else {
                phi_366_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e238_ = phi_366_;
            let _e239_ = (_e218_ + 3u);
            let _e240_ = (_e239_ < _e109_);
            if _e240_ {
                if _e240_ {
                } else {
                    break;
                }
                let _e243_ = quad_params.member[_e239_];
                phi_379_ = _e243_;
            } else {
                phi_379_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e245_ = phi_379_;
            let _e246_ = (_e218_ + 4u);
            let _e247_ = (_e246_ < _e109_);
            if _e247_ {
                if _e247_ {
                } else {
                    break;
                }
                let _e250_ = quad_params.member[_e246_];
                phi_392_ = _e250_;
            } else {
                phi_392_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e252_ = phi_392_;
            let _e253_ = (_e218_ + 5u);
            let _e254_ = (_e253_ < _e109_);
            if _e254_ {
                if _e254_ {
                } else {
                    break;
                }
                let _e257_ = quad_params.member[_e253_];
                phi_405_ = _e257_;
            } else {
                phi_405_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e259_ = phi_405_;
            let _e260_ = (_e218_ + 6u);
            let _e261_ = (_e260_ < _e109_);
            if _e261_ {
                if _e261_ {
                } else {
                    break;
                }
                let _e264_ = quad_params.member[_e260_];
                phi_418_ = _e264_;
            } else {
                phi_418_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e266_ = phi_418_;
            let _e267_ = (_e218_ + 7u);
            let _e268_ = (_e267_ < _e109_);
            if _e268_ {
                if _e268_ {
                } else {
                    break;
                }
                let _e271_ = quad_params.member[_e267_];
                phi_431_ = _e271_;
            } else {
                phi_431_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e273_ = phi_431_;
            let _e274_ = (_e218_ + 8u);
            let _e275_ = (_e274_ < _e109_);
            if _e275_ {
                if _e275_ {
                } else {
                    break;
                }
                let _e278_ = quad_params.member[_e274_];
                phi_444_ = _e278_;
            } else {
                phi_444_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e280_ = phi_444_;
            let _e281_ = (_e218_ + 9u);
            let _e282_ = (_e281_ < _e109_);
            if _e282_ {
                if _e282_ {
                } else {
                    break;
                }
                let _e285_ = quad_params.member[_e281_];
                phi_457_ = _e285_;
            } else {
                phi_457_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e287_ = phi_457_;
            let _e288_ = (_e218_ + 10u);
            let _e289_ = (_e288_ < _e109_);
            if _e289_ {
                if _e289_ {
                } else {
                    break;
                }
                let _e292_ = quad_params.member[_e288_];
                phi_470_ = _e292_;
            } else {
                phi_470_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e294_ = phi_470_;
            let _e295_ = (_e218_ + 11u);
            let _e296_ = (_e295_ < _e109_);
            if _e296_ {
                if _e296_ {
                } else {
                    break;
                }
                let _e299_ = quad_params.member[_e295_];
                phi_483_ = _e299_;
            } else {
                phi_483_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e301_ = phi_483_;
            let _e302_ = (_e218_ + 12u);
            let _e303_ = (_e302_ < _e109_);
            if _e303_ {
                if _e303_ {
                } else {
                    break;
                }
                let _e306_ = quad_params.member[_e302_];
                phi_496_ = _e306_;
            } else {
                phi_496_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e308_ = phi_496_;
            let _e309_ = (_e218_ + 13u);
            let _e310_ = (_e309_ < _e109_);
            if _e310_ {
                if _e310_ {
                } else {
                    break;
                }
                let _e313_ = quad_params.member[_e309_];
                phi_509_ = _e313_;
            } else {
                phi_509_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e315_ = phi_509_;
            let _e316_ = (_e218_ + 14u);
            let _e317_ = (_e316_ < _e109_);
            if _e317_ {
                if _e317_ {
                } else {
                    break;
                }
                let _e320_ = quad_params.member[_e316_];
                phi_522_ = _e320_;
            } else {
                phi_522_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e322_ = phi_522_;
            let _e323_ = (_e218_ + 15u);
            let _e324_ = (_e323_ < _e109_);
            if _e324_ {
                if _e324_ {
                } else {
                    break;
                }
                let _e327_ = quad_params.member[_e323_];
                phi_535_ = _e327_;
            } else {
                phi_535_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e329_ = phi_535_;
            let _e330_ = (_e218_ + 16u);
            let _e331_ = (_e330_ < _e109_);
            if _e331_ {
                if _e331_ {
                } else {
                    break;
                }
                let _e334_ = quad_params.member[_e330_];
                phi_548_ = _e334_;
            } else {
                phi_548_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e336_ = phi_548_;
            let _e337_ = (_e218_ + 17u);
            let _e338_ = (_e337_ < _e109_);
            if _e338_ {
                if _e338_ {
                } else {
                    break;
                }
                let _e341_ = quad_params.member[_e337_];
                phi_561_ = _e341_;
            } else {
                phi_561_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e343_ = phi_561_;
            let _e344_ = (_e218_ + 18u);
            let _e345_ = (_e344_ < _e109_);
            if _e345_ {
                if _e345_ {
                } else {
                    break;
                }
                let _e348_ = quad_params.member[_e344_];
                phi_574_ = _e348_;
            } else {
                phi_574_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e350_ = phi_574_;
            let _e351_ = (_e218_ + 19u);
            let _e352_ = (_e351_ < _e109_);
            if _e352_ {
                if _e352_ {
                } else {
                    break;
                }
                let _e355_ = quad_params.member[_e351_];
                phi_587_ = _e355_;
            } else {
                phi_587_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e357_ = phi_587_;
            let _e358_ = (_e218_ + 20u);
            let _e359_ = (_e358_ < _e109_);
            if _e359_ {
                if _e359_ {
                } else {
                    break;
                }
                let _e362_ = quad_params.member[_e358_];
                phi_600_ = _e362_;
            } else {
                phi_600_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e364_ = phi_600_;
            let _e365_ = (_e218_ + 21u);
            let _e366_ = (_e365_ < _e109_);
            if _e366_ {
                if _e366_ {
                } else {
                    break;
                }
                let _e369_ = quad_params.member[_e365_];
                phi_613_ = _e369_;
            } else {
                phi_613_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e371_ = phi_613_;
            let _e372_ = (_e218_ + 22u);
            let _e373_ = (_e372_ < _e109_);
            if _e373_ {
                if _e373_ {
                } else {
                    break;
                }
                let _e376_ = quad_params.member[_e372_];
                phi_626_ = _e376_;
            } else {
                phi_626_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e378_ = phi_626_;
            let _e380_ = ((_e218_ + 23u) < _e109_);
            if _e380_ {
                if _e380_ {
                } else {
                    break;
                }
            }
            let _e382_ = bitcast<u32>(_e371_.x);
            let _e383_ = (_e382_ & 2139095040u);
            if (_e383_ != 2139095040u) {
                if (_e371_.x <= 0f) {
                    phi_673_ = 0f;
                } else {
                    if (_e371_.x >= 1f) {
                        phi_672_ = 1f;
                    } else {
                        if (_e383_ == 2139095040u) {
                            phi_666_ = true;
                        } else {
                            phi_666_ = (_e383_ == 0u);
                        }
                        let _e392_ = phi_666_;
                        phi_672_ = bitcast<f32>(select(_e382_, 0u, _e392_));
                    }
                    let _e396_ = phi_672_;
                    phi_673_ = _e396_;
                }
                let _e398_ = phi_673_;
                phi_680_ = _e398_;
            } else {
                phi_680_ = select(0f, 1f, (_e382_ == 2139095040u));
            }
            let _e400_ = phi_680_;
            let _e402_ = bitcast<u32>(_e371_.y);
            let _e403_ = (_e402_ & 2139095040u);
            if (_e403_ != 2139095040u) {
                if (_e371_.y <= 0f) {
                    phi_713_ = 0f;
                } else {
                    if (_e371_.y >= 1f) {
                        phi_712_ = 1f;
                    } else {
                        if (_e403_ == 2139095040u) {
                            phi_706_ = true;
                        } else {
                            phi_706_ = (_e403_ == 0u);
                        }
                        let _e412_ = phi_706_;
                        phi_712_ = bitcast<f32>(select(_e402_, 0u, _e412_));
                    }
                    let _e416_ = phi_712_;
                    phi_713_ = _e416_;
                }
                let _e418_ = phi_713_;
                phi_720_ = _e418_;
            } else {
                phi_720_ = select(0f, 1f, (_e402_ == 2139095040u));
            }
            let _e420_ = phi_720_;
            let _e422_ = bitcast<u32>(_e371_.z);
            let _e423_ = (_e422_ & 2139095040u);
            if (_e423_ != 2139095040u) {
                if (_e371_.z <= 0f) {
                    phi_753_ = 0f;
                } else {
                    if (_e371_.z >= 1f) {
                        phi_752_ = 1f;
                    } else {
                        if (_e423_ == 2139095040u) {
                            phi_746_ = true;
                        } else {
                            phi_746_ = (_e423_ == 0u);
                        }
                        let _e432_ = phi_746_;
                        phi_752_ = bitcast<f32>(select(_e422_, 0u, _e432_));
                    }
                    let _e436_ = phi_752_;
                    phi_753_ = _e436_;
                }
                let _e438_ = phi_753_;
                phi_760_ = _e438_;
            } else {
                phi_760_ = select(0f, 1f, (_e422_ == 2139095040u));
            }
            let _e440_ = phi_760_;
            let _e442_ = bitcast<u32>(_e371_.w);
            let _e443_ = (_e442_ & 2139095040u);
            if (_e443_ != 2139095040u) {
                if (_e371_.w <= 0f) {
                    phi_793_ = 0f;
                } else {
                    if (_e371_.w >= 1f) {
                        phi_792_ = 1f;
                    } else {
                        if (_e443_ == 2139095040u) {
                            phi_786_ = true;
                        } else {
                            phi_786_ = (_e443_ == 0u);
                        }
                        let _e452_ = phi_786_;
                        phi_792_ = bitcast<f32>(select(_e442_, 0u, _e452_));
                    }
                    let _e456_ = phi_792_;
                    phi_793_ = _e456_;
                }
                let _e458_ = phi_793_;
                phi_800_ = _e458_;
            } else {
                phi_800_ = select(0f, 1f, (_e442_ == 2139095040u));
            }
            let _e460_ = phi_800_;
            if _e148_ {
                let _e465_ = (_e147_ > 0u);
                if _e465_ {
                    let _e466_ = (_e147_ < _e140_);
                    if _e466_ {
                        let _e469_ = (((_e125_ * _e140_) + _e147_) * 8u);
                        let _e470_ = (_e469_ < _e111_);
                        if _e470_ {
                            if _e470_ {
                            } else {
                                break;
                            }
                            let _e473_ = tile_checkpoints.member[_e469_];
                            phi_832_ = _e473_;
                        } else {
                            phi_832_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e475_ = phi_832_;
                        let _e477_ = bitcast<u32>(_e475_.x);
                        let _e478_ = (_e477_ & 2139095040u);
                        if (_e478_ == 2139095040u) {
                            phi_844_ = true;
                        } else {
                            phi_844_ = (_e478_ == 0u);
                        }
                        let _e482_ = phi_844_;
                        let _e486_ = bitcast<u32>(_e475_.y);
                        let _e487_ = (_e486_ & 2139095040u);
                        if (_e487_ == 2139095040u) {
                            phi_861_ = true;
                        } else {
                            phi_861_ = (_e487_ == 0u);
                        }
                        let _e491_ = phi_861_;
                        let _e495_ = bitcast<u32>(_e475_.z);
                        let _e496_ = (_e495_ & 2139095040u);
                        if (_e496_ == 2139095040u) {
                            phi_878_ = true;
                        } else {
                            phi_878_ = (_e496_ == 0u);
                        }
                        let _e500_ = phi_878_;
                        let _e504_ = bitcast<u32>(_e475_.w);
                        let _e505_ = (_e504_ & 2139095040u);
                        if (_e505_ == 2139095040u) {
                            phi_895_ = true;
                        } else {
                            phi_895_ = (_e505_ == 0u);
                        }
                        let _e509_ = phi_895_;
                        let _e513_ = (_e469_ + 1u);
                        let _e514_ = (_e513_ < _e111_);
                        if _e514_ {
                            if _e514_ {
                            } else {
                                break;
                            }
                            let _e517_ = tile_checkpoints.member[_e513_];
                            phi_914_ = _e517_;
                        } else {
                            phi_914_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e519_ = phi_914_;
                        let _e521_ = bitcast<u32>(_e519_.x);
                        let _e522_ = (_e521_ & 2139095040u);
                        if (_e522_ == 2139095040u) {
                            phi_926_ = true;
                        } else {
                            phi_926_ = (_e522_ == 0u);
                        }
                        let _e526_ = phi_926_;
                        let _e530_ = bitcast<u32>(_e519_.y);
                        let _e531_ = (_e530_ & 2139095040u);
                        if (_e531_ == 2139095040u) {
                            phi_943_ = true;
                        } else {
                            phi_943_ = (_e531_ == 0u);
                        }
                        let _e535_ = phi_943_;
                        let _e539_ = bitcast<u32>(_e519_.z);
                        let _e540_ = (_e539_ & 2139095040u);
                        if (_e540_ == 2139095040u) {
                            phi_960_ = true;
                        } else {
                            phi_960_ = (_e540_ == 0u);
                        }
                        let _e544_ = phi_960_;
                        let _e548_ = bitcast<u32>(_e519_.w);
                        let _e549_ = (_e548_ & 2139095040u);
                        if (_e549_ == 2139095040u) {
                            phi_977_ = true;
                        } else {
                            phi_977_ = (_e549_ == 0u);
                        }
                        let _e553_ = phi_977_;
                        let _e557_ = (_e469_ + 2u);
                        let _e558_ = (_e557_ < _e111_);
                        if _e558_ {
                            if _e558_ {
                            } else {
                                break;
                            }
                            let _e561_ = tile_checkpoints.member[_e557_];
                            phi_996_ = _e561_;
                        } else {
                            phi_996_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e563_ = phi_996_;
                        let _e565_ = bitcast<u32>(_e563_.x);
                        let _e566_ = (_e565_ & 2139095040u);
                        if (_e566_ == 2139095040u) {
                            phi_1008_ = true;
                        } else {
                            phi_1008_ = (_e566_ == 0u);
                        }
                        let _e570_ = phi_1008_;
                        let _e574_ = bitcast<u32>(_e563_.y);
                        let _e575_ = (_e574_ & 2139095040u);
                        if (_e575_ == 2139095040u) {
                            phi_1025_ = true;
                        } else {
                            phi_1025_ = (_e575_ == 0u);
                        }
                        let _e579_ = phi_1025_;
                        let _e583_ = bitcast<u32>(_e563_.z);
                        let _e584_ = (_e583_ & 2139095040u);
                        if (_e584_ == 2139095040u) {
                            phi_1042_ = true;
                        } else {
                            phi_1042_ = (_e584_ == 0u);
                        }
                        let _e588_ = phi_1042_;
                        let _e592_ = bitcast<u32>(_e563_.w);
                        let _e593_ = (_e592_ & 2139095040u);
                        if (_e593_ == 2139095040u) {
                            phi_1059_ = true;
                        } else {
                            phi_1059_ = (_e593_ == 0u);
                        }
                        let _e597_ = phi_1059_;
                        let _e601_ = (_e469_ + 3u);
                        let _e602_ = (_e601_ < _e111_);
                        if _e602_ {
                            if _e602_ {
                            } else {
                                break;
                            }
                            let _e605_ = tile_checkpoints.member[_e601_];
                            phi_1078_ = _e605_;
                        } else {
                            phi_1078_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e607_ = phi_1078_;
                        let _e609_ = bitcast<u32>(_e607_.x);
                        let _e610_ = (_e609_ & 2139095040u);
                        if (_e610_ == 2139095040u) {
                            phi_1090_ = true;
                        } else {
                            phi_1090_ = (_e610_ == 0u);
                        }
                        let _e614_ = phi_1090_;
                        let _e618_ = bitcast<u32>(_e607_.y);
                        let _e619_ = (_e618_ & 2139095040u);
                        if (_e619_ == 2139095040u) {
                            phi_1107_ = true;
                        } else {
                            phi_1107_ = (_e619_ == 0u);
                        }
                        let _e623_ = phi_1107_;
                        let _e627_ = bitcast<u32>(_e607_.z);
                        let _e628_ = (_e627_ & 2139095040u);
                        if (_e628_ == 2139095040u) {
                            phi_1124_ = true;
                        } else {
                            phi_1124_ = (_e628_ == 0u);
                        }
                        let _e632_ = phi_1124_;
                        let _e636_ = bitcast<u32>(_e607_.w);
                        let _e637_ = (_e636_ & 2139095040u);
                        if (_e637_ == 2139095040u) {
                            phi_1141_ = true;
                        } else {
                            phi_1141_ = (_e637_ == 0u);
                        }
                        let _e641_ = phi_1141_;
                        let _e645_ = (_e469_ + 4u);
                        let _e646_ = (_e645_ < _e111_);
                        if _e646_ {
                            if _e646_ {
                            } else {
                                break;
                            }
                            let _e649_ = tile_checkpoints.member[_e645_];
                            phi_1160_ = _e649_;
                        } else {
                            phi_1160_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e651_ = phi_1160_;
                        let _e653_ = bitcast<u32>(_e651_.x);
                        let _e654_ = (_e653_ & 2139095040u);
                        if (_e654_ == 2139095040u) {
                            phi_1172_ = true;
                        } else {
                            phi_1172_ = (_e654_ == 0u);
                        }
                        let _e658_ = phi_1172_;
                        let _e662_ = bitcast<u32>(_e651_.y);
                        let _e663_ = (_e662_ & 2139095040u);
                        if (_e663_ == 2139095040u) {
                            phi_1189_ = true;
                        } else {
                            phi_1189_ = (_e663_ == 0u);
                        }
                        let _e667_ = phi_1189_;
                        let _e671_ = bitcast<u32>(_e651_.z);
                        let _e672_ = (_e671_ & 2139095040u);
                        if (_e672_ == 2139095040u) {
                            phi_1206_ = true;
                        } else {
                            phi_1206_ = (_e672_ == 0u);
                        }
                        let _e676_ = phi_1206_;
                        let _e680_ = bitcast<u32>(_e651_.w);
                        let _e681_ = (_e680_ & 2139095040u);
                        if (_e681_ == 2139095040u) {
                            phi_1223_ = true;
                        } else {
                            phi_1223_ = (_e681_ == 0u);
                        }
                        let _e685_ = phi_1223_;
                        let _e689_ = (_e469_ + 5u);
                        let _e690_ = (_e689_ < _e111_);
                        if _e690_ {
                            if _e690_ {
                            } else {
                                break;
                            }
                            let _e693_ = tile_checkpoints.member[_e689_];
                            phi_1242_ = _e693_;
                        } else {
                            phi_1242_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e695_ = phi_1242_;
                        let _e697_ = bitcast<u32>(_e695_.x);
                        let _e698_ = (_e697_ & 2139095040u);
                        if (_e698_ == 2139095040u) {
                            phi_1254_ = true;
                        } else {
                            phi_1254_ = (_e698_ == 0u);
                        }
                        let _e702_ = phi_1254_;
                        let _e706_ = bitcast<u32>(_e695_.y);
                        let _e707_ = (_e706_ & 2139095040u);
                        if (_e707_ == 2139095040u) {
                            phi_1271_ = true;
                        } else {
                            phi_1271_ = (_e707_ == 0u);
                        }
                        let _e711_ = phi_1271_;
                        let _e715_ = bitcast<u32>(_e695_.z);
                        let _e716_ = (_e715_ & 2139095040u);
                        if (_e716_ == 2139095040u) {
                            phi_1288_ = true;
                        } else {
                            phi_1288_ = (_e716_ == 0u);
                        }
                        let _e720_ = phi_1288_;
                        let _e724_ = bitcast<u32>(_e695_.w);
                        let _e725_ = (_e724_ & 2139095040u);
                        if (_e725_ == 2139095040u) {
                            phi_1305_ = true;
                        } else {
                            phi_1305_ = (_e725_ == 0u);
                        }
                        let _e729_ = phi_1305_;
                        let _e733_ = (_e469_ + 6u);
                        let _e734_ = (_e733_ < _e111_);
                        if _e734_ {
                            if _e734_ {
                            } else {
                                break;
                            }
                            let _e737_ = tile_checkpoints.member[_e733_];
                            phi_1324_ = _e737_;
                        } else {
                            phi_1324_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e739_ = phi_1324_;
                        let _e741_ = bitcast<u32>(_e739_.x);
                        let _e742_ = (_e741_ & 2139095040u);
                        if (_e742_ == 2139095040u) {
                            phi_1336_ = true;
                        } else {
                            phi_1336_ = (_e742_ == 0u);
                        }
                        let _e746_ = phi_1336_;
                        let _e748_ = bitcast<f32>(select(_e741_, 0u, _e746_));
                        if (_e748_ < -1000000000000000000f) {
                            phi_1353_ = -1000000000000000000f;
                        } else {
                            phi_1353_ = select(_e748_, 1000000000000000000f, (_e748_ > 1000000000000000000f));
                        }
                        let _e753_ = phi_1353_;
                        let _e755_ = bitcast<u32>(_e739_.y);
                        let _e756_ = (_e755_ & 2139095040u);
                        if (_e756_ == 2139095040u) {
                            phi_1365_ = true;
                        } else {
                            phi_1365_ = (_e756_ == 0u);
                        }
                        let _e760_ = phi_1365_;
                        let _e762_ = bitcast<f32>(select(_e755_, 0u, _e760_));
                        if (_e762_ < -1000000000000000000f) {
                            phi_1382_ = -1000000000000000000f;
                        } else {
                            phi_1382_ = select(_e762_, 1000000000000000000f, (_e762_ > 1000000000000000000f));
                        }
                        let _e767_ = phi_1382_;
                        let _e769_ = bitcast<u32>(_e739_.z);
                        let _e770_ = (_e769_ & 2139095040u);
                        if (_e770_ == 2139095040u) {
                            phi_1394_ = true;
                        } else {
                            phi_1394_ = (_e770_ == 0u);
                        }
                        let _e774_ = phi_1394_;
                        let _e776_ = bitcast<f32>(select(_e769_, 0u, _e774_));
                        if (_e776_ < -1000000000000000000f) {
                            phi_1411_ = -1000000000000000000f;
                        } else {
                            phi_1411_ = select(_e776_, 1000000000000000000f, (_e776_ > 1000000000000000000f));
                        }
                        let _e781_ = phi_1411_;
                        let _e783_ = bitcast<u32>(_e739_.w);
                        let _e784_ = (_e783_ & 2139095040u);
                        if (_e784_ == 2139095040u) {
                            phi_1423_ = true;
                        } else {
                            phi_1423_ = (_e784_ == 0u);
                        }
                        let _e788_ = phi_1423_;
                        let _e790_ = bitcast<f32>(select(_e783_, 0u, _e788_));
                        if (_e790_ < -1000000000000000000f) {
                            phi_1440_ = -1000000000000000000f;
                        } else {
                            phi_1440_ = select(_e790_, 1000000000000000000f, (_e790_ > 1000000000000000000f));
                        }
                        let _e795_ = phi_1440_;
                        let _e797_ = (_e469_ + 7u);
                        let _e798_ = (_e797_ < _e111_);
                        if _e798_ {
                            if _e798_ {
                            } else {
                                break;
                            }
                            let _e801_ = tile_checkpoints.member[_e797_];
                            phi_1454_ = _e801_;
                        } else {
                            phi_1454_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e803_ = phi_1454_;
                        let _e805_ = bitcast<u32>(_e803_.x);
                        let _e806_ = (_e805_ & 2139095040u);
                        if (_e806_ == 2139095040u) {
                            phi_1466_ = true;
                        } else {
                            phi_1466_ = (_e806_ == 0u);
                        }
                        let _e810_ = phi_1466_;
                        let _e812_ = bitcast<f32>(select(_e805_, 0u, _e810_));
                        if (_e812_ < -1000000000000000000f) {
                            phi_1483_ = -1000000000000000000f;
                        } else {
                            phi_1483_ = select(_e812_, 1000000000000000000f, (_e812_ > 1000000000000000000f));
                        }
                        let _e817_ = phi_1483_;
                        let _e819_ = bitcast<u32>(_e803_.y);
                        let _e820_ = (_e819_ & 2139095040u);
                        if (_e820_ == 2139095040u) {
                            phi_1495_ = true;
                        } else {
                            phi_1495_ = (_e820_ == 0u);
                        }
                        let _e824_ = phi_1495_;
                        let _e826_ = bitcast<f32>(select(_e819_, 0u, _e824_));
                        if (_e826_ < -1000000000000000000f) {
                            phi_1512_ = -1000000000000000000f;
                        } else {
                            phi_1512_ = select(_e826_, 1000000000000000000f, (_e826_ > 1000000000000000000f));
                        }
                        let _e831_ = phi_1512_;
                        let _e833_ = bitcast<u32>(_e803_.z);
                        let _e834_ = (_e833_ & 2139095040u);
                        if (_e834_ == 2139095040u) {
                            phi_1524_ = true;
                        } else {
                            phi_1524_ = (_e834_ == 0u);
                        }
                        let _e838_ = phi_1524_;
                        let _e840_ = bitcast<f32>(select(_e833_, 0u, _e838_));
                        if (_e840_ < -1000000000000000000f) {
                            phi_1541_ = -1000000000000000000f;
                        } else {
                            phi_1541_ = select(_e840_, 1000000000000000000f, (_e840_ > 1000000000000000000f));
                        }
                        let _e845_ = phi_1541_;
                        let _e847_ = bitcast<u32>(_e803_.w);
                        let _e848_ = (_e847_ & 2139095040u);
                        if (_e848_ == 2139095040u) {
                            phi_1553_ = true;
                        } else {
                            phi_1553_ = (_e848_ == 0u);
                        }
                        let _e852_ = phi_1553_;
                        let _e854_ = bitcast<f32>(select(_e847_, 0u, _e852_));
                        if (_e854_ < -1000000000000000000f) {
                            phi_1570_ = -1000000000000000000f;
                        } else {
                            phi_1570_ = select(_e854_, 1000000000000000000f, (_e854_ > 1000000000000000000f));
                        }
                        let _e859_ = phi_1570_;
                        phi_1581_ = type_32_(type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e477_, 0u, _e482_)), bitcast<f32>(select(_e486_, 0u, _e491_)), bitcast<f32>(select(_e495_, 0u, _e500_)), bitcast<f32>(select(_e504_, 0u, _e509_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e521_, 0u, _e526_)), bitcast<f32>(select(_e530_, 0u, _e535_)), bitcast<f32>(select(_e539_, 0u, _e544_)), bitcast<f32>(select(_e548_, 0u, _e553_))))), type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e565_, 0u, _e570_)), bitcast<f32>(select(_e574_, 0u, _e579_)), bitcast<f32>(select(_e583_, 0u, _e588_)), bitcast<f32>(select(_e592_, 0u, _e597_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e609_, 0u, _e614_)), bitcast<f32>(select(_e618_, 0u, _e623_)), bitcast<f32>(select(_e627_, 0u, _e632_)), bitcast<f32>(select(_e636_, 0u, _e641_))))), type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e653_, 0u, _e658_)), bitcast<f32>(select(_e662_, 0u, _e667_)), bitcast<f32>(select(_e671_, 0u, _e676_)), bitcast<f32>(select(_e680_, 0u, _e685_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e697_, 0u, _e702_)), bitcast<f32>(select(_e706_, 0u, _e711_)), bitcast<f32>(select(_e715_, 0u, _e720_)), bitcast<f32>(select(_e724_, 0u, _e729_))))), type_27_(type_24_(array<f32, 4>(_e753_, _e767_, _e781_, _e795_)), type_24_(array<f32, 4>(_e817_, _e831_, _e845_, _e859_))));
                    } else {
                        phi_1581_ = type_32_();
                    }
                    let _e875_ = phi_1581_;
                    phi_1583_ = _e875_;
                    phi_1584_ = _e466_;
                    phi_1587_ = select(true, false, _e466_);
                } else {
                    phi_1583_ = type_32_();
                    phi_1584_ = bool();
                    phi_1587_ = false;
                }
                let _e878_ = phi_1583_;
                let _e880_ = phi_1584_;
                let _e882_ = phi_1587_;
                phi_1588_ = _e878_;
                phi_1589_ = _e880_;
                phi_1591_ = select(true, false, _e465_);
                phi_1592_ = _e882_;
            } else {
                phi_1588_ = type_32_();
                phi_1589_ = bool();
                phi_1591_ = true;
                phi_1592_ = false;
            }
            let _e885_ = phi_1588_;
            let _e887_ = phi_1589_;
            let _e889_ = phi_1591_;
            let _e891_ = phi_1592_;
            let _e893_ = select(_e891_, true, _e889_);
            if _e893_ {
                phi_1614_ = type_32_(type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))));
            } else {
                phi_1614_ = _e885_;
            }
            let _e895_ = phi_1614_;
            phi_16480_ = false;
            phi_1618_ = type_16_(_e215_, _e217_);
            phi_1621_ = _e895_;
            loop {
                let _e899_ = phi_16480_;
                let _e901_ = phi_1618_;
                let _e903_ = phi_1621_;
                if (_e901_.member < _e901_.member_1_) {
                    phi_1619_ = type_16_((_e901_.member + 1u), _e901_.member_1_);
                    phi_1644_ = type_16_(1u, _e901_.member);
                } else {
                    phi_1619_ = _e901_;
                    phi_1644_ = type_16_(0u, u32());
                }
                let _e916_ = phi_1619_;
                let _e918_ = phi_1644_;
                let _e922_ = (bitcast<i32>(_e918_.member) != 0i);
                if _e922_ {
                    let _e923_ = (_e918_.member_1_ * 256u);
                    if (_e128_ > _e923_) {
                        phi_1658_ = (_e128_ - _e923_);
                    } else {
                        phi_1658_ = 0u;
                    }
                    let _e927_ = phi_1658_;
                    let _e929_ = select(_e927_, 256u, (256u < _e927_));
                    let _e931_ = (_e114_ < _e929_);
                    let _e932_ = (_e114_ == 0u);
                    if _e932_ {
                        if select(_e887_, select(_e887_, false, _e889_), _e893_) {
                            phi_1678_ = select(true, false, (_e918_.member_1_ == _e147_));
                        } else {
                            phi_1678_ = true;
                        }
                        let _e936_ = phi_1678_;
                        if _e936_ {
                            let _e939_ = (((_e125_ * _e140_) + _e918_.member_1_) * 8u);
                            let _e964_ = bitcast<u32>(_e903_.member.member.member[0]);
                            let _e965_ = (_e964_ & 2139095040u);
                            if (_e965_ == 2139095040u) {
                                phi_1727_ = true;
                            } else {
                                phi_1727_ = (_e965_ == 0u);
                            }
                            let _e969_ = phi_1727_;
                            let _e972_ = bitcast<u32>(_e903_.member.member.member[1]);
                            let _e973_ = (_e972_ & 2139095040u);
                            if (_e973_ == 2139095040u) {
                                phi_1743_ = true;
                            } else {
                                phi_1743_ = (_e973_ == 0u);
                            }
                            let _e977_ = phi_1743_;
                            let _e980_ = bitcast<u32>(_e903_.member.member.member[2]);
                            let _e981_ = (_e980_ & 2139095040u);
                            if (_e981_ == 2139095040u) {
                                phi_1759_ = true;
                            } else {
                                phi_1759_ = (_e981_ == 0u);
                            }
                            let _e985_ = phi_1759_;
                            let _e988_ = bitcast<u32>(_e903_.member.member.member[3]);
                            let _e989_ = (_e988_ & 2139095040u);
                            if (_e989_ == 2139095040u) {
                                phi_1775_ = true;
                            } else {
                                phi_1775_ = (_e989_ == 0u);
                            }
                            let _e993_ = phi_1775_;
                            let _e997_ = (_e939_ < _e111_);
                            if _e997_ {
                                if _e997_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e939_] = vec4<f32>(bitcast<f32>(select(_e964_, 0u, _e969_)), bitcast<f32>(select(_e972_, 0u, _e977_)), bitcast<f32>(select(_e980_, 0u, _e985_)), bitcast<f32>(select(_e988_, 0u, _e993_)));
                            }
                            let _e1000_ = (_e939_ + 1u);
                            let _e1005_ = bitcast<u32>(_e903_.member.member_1_.member[0]);
                            let _e1006_ = (_e1005_ & 2139095040u);
                            if (_e1006_ == 2139095040u) {
                                phi_1807_ = true;
                            } else {
                                phi_1807_ = (_e1006_ == 0u);
                            }
                            let _e1010_ = phi_1807_;
                            let _e1013_ = bitcast<u32>(_e903_.member.member_1_.member[1]);
                            let _e1014_ = (_e1013_ & 2139095040u);
                            if (_e1014_ == 2139095040u) {
                                phi_1823_ = true;
                            } else {
                                phi_1823_ = (_e1014_ == 0u);
                            }
                            let _e1018_ = phi_1823_;
                            let _e1021_ = bitcast<u32>(_e903_.member.member_1_.member[2]);
                            let _e1022_ = (_e1021_ & 2139095040u);
                            if (_e1022_ == 2139095040u) {
                                phi_1839_ = true;
                            } else {
                                phi_1839_ = (_e1022_ == 0u);
                            }
                            let _e1026_ = phi_1839_;
                            let _e1029_ = bitcast<u32>(_e903_.member.member_1_.member[3]);
                            let _e1030_ = (_e1029_ & 2139095040u);
                            if (_e1030_ == 2139095040u) {
                                phi_1855_ = true;
                            } else {
                                phi_1855_ = (_e1030_ == 0u);
                            }
                            let _e1034_ = phi_1855_;
                            let _e1038_ = (_e1000_ < _e111_);
                            if _e1038_ {
                                if _e1038_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1000_] = vec4<f32>(bitcast<f32>(select(_e1005_, 0u, _e1010_)), bitcast<f32>(select(_e1013_, 0u, _e1018_)), bitcast<f32>(select(_e1021_, 0u, _e1026_)), bitcast<f32>(select(_e1029_, 0u, _e1034_)));
                            }
                            let _e1041_ = (_e939_ + 2u);
                            let _e1046_ = bitcast<u32>(_e903_.member_1_.member.member[0]);
                            let _e1047_ = (_e1046_ & 2139095040u);
                            if (_e1047_ == 2139095040u) {
                                phi_1887_ = true;
                            } else {
                                phi_1887_ = (_e1047_ == 0u);
                            }
                            let _e1051_ = phi_1887_;
                            let _e1054_ = bitcast<u32>(_e903_.member_1_.member.member[1]);
                            let _e1055_ = (_e1054_ & 2139095040u);
                            if (_e1055_ == 2139095040u) {
                                phi_1903_ = true;
                            } else {
                                phi_1903_ = (_e1055_ == 0u);
                            }
                            let _e1059_ = phi_1903_;
                            let _e1062_ = bitcast<u32>(_e903_.member_1_.member.member[2]);
                            let _e1063_ = (_e1062_ & 2139095040u);
                            if (_e1063_ == 2139095040u) {
                                phi_1919_ = true;
                            } else {
                                phi_1919_ = (_e1063_ == 0u);
                            }
                            let _e1067_ = phi_1919_;
                            let _e1070_ = bitcast<u32>(_e903_.member_1_.member.member[3]);
                            let _e1071_ = (_e1070_ & 2139095040u);
                            if (_e1071_ == 2139095040u) {
                                phi_1935_ = true;
                            } else {
                                phi_1935_ = (_e1071_ == 0u);
                            }
                            let _e1075_ = phi_1935_;
                            let _e1079_ = (_e1041_ < _e111_);
                            if _e1079_ {
                                if _e1079_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1041_] = vec4<f32>(bitcast<f32>(select(_e1046_, 0u, _e1051_)), bitcast<f32>(select(_e1054_, 0u, _e1059_)), bitcast<f32>(select(_e1062_, 0u, _e1067_)), bitcast<f32>(select(_e1070_, 0u, _e1075_)));
                            }
                            let _e1082_ = (_e939_ + 3u);
                            let _e1087_ = bitcast<u32>(_e903_.member_1_.member_1_.member[0]);
                            let _e1088_ = (_e1087_ & 2139095040u);
                            if (_e1088_ == 2139095040u) {
                                phi_1967_ = true;
                            } else {
                                phi_1967_ = (_e1088_ == 0u);
                            }
                            let _e1092_ = phi_1967_;
                            let _e1095_ = bitcast<u32>(_e903_.member_1_.member_1_.member[1]);
                            let _e1096_ = (_e1095_ & 2139095040u);
                            if (_e1096_ == 2139095040u) {
                                phi_1983_ = true;
                            } else {
                                phi_1983_ = (_e1096_ == 0u);
                            }
                            let _e1100_ = phi_1983_;
                            let _e1103_ = bitcast<u32>(_e903_.member_1_.member_1_.member[2]);
                            let _e1104_ = (_e1103_ & 2139095040u);
                            if (_e1104_ == 2139095040u) {
                                phi_1999_ = true;
                            } else {
                                phi_1999_ = (_e1104_ == 0u);
                            }
                            let _e1108_ = phi_1999_;
                            let _e1111_ = bitcast<u32>(_e903_.member_1_.member_1_.member[3]);
                            let _e1112_ = (_e1111_ & 2139095040u);
                            if (_e1112_ == 2139095040u) {
                                phi_2015_ = true;
                            } else {
                                phi_2015_ = (_e1112_ == 0u);
                            }
                            let _e1116_ = phi_2015_;
                            let _e1120_ = (_e1082_ < _e111_);
                            if _e1120_ {
                                if _e1120_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1082_] = vec4<f32>(bitcast<f32>(select(_e1087_, 0u, _e1092_)), bitcast<f32>(select(_e1095_, 0u, _e1100_)), bitcast<f32>(select(_e1103_, 0u, _e1108_)), bitcast<f32>(select(_e1111_, 0u, _e1116_)));
                            }
                            let _e1123_ = (_e939_ + 4u);
                            let _e1128_ = bitcast<u32>(_e903_.member_2_.member.member[0]);
                            let _e1129_ = (_e1128_ & 2139095040u);
                            if (_e1129_ == 2139095040u) {
                                phi_2047_ = true;
                            } else {
                                phi_2047_ = (_e1129_ == 0u);
                            }
                            let _e1133_ = phi_2047_;
                            let _e1136_ = bitcast<u32>(_e903_.member_2_.member.member[1]);
                            let _e1137_ = (_e1136_ & 2139095040u);
                            if (_e1137_ == 2139095040u) {
                                phi_2063_ = true;
                            } else {
                                phi_2063_ = (_e1137_ == 0u);
                            }
                            let _e1141_ = phi_2063_;
                            let _e1144_ = bitcast<u32>(_e903_.member_2_.member.member[2]);
                            let _e1145_ = (_e1144_ & 2139095040u);
                            if (_e1145_ == 2139095040u) {
                                phi_2079_ = true;
                            } else {
                                phi_2079_ = (_e1145_ == 0u);
                            }
                            let _e1149_ = phi_2079_;
                            let _e1152_ = bitcast<u32>(_e903_.member_2_.member.member[3]);
                            let _e1153_ = (_e1152_ & 2139095040u);
                            if (_e1153_ == 2139095040u) {
                                phi_2095_ = true;
                            } else {
                                phi_2095_ = (_e1153_ == 0u);
                            }
                            let _e1157_ = phi_2095_;
                            let _e1161_ = (_e1123_ < _e111_);
                            if _e1161_ {
                                if _e1161_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1123_] = vec4<f32>(bitcast<f32>(select(_e1128_, 0u, _e1133_)), bitcast<f32>(select(_e1136_, 0u, _e1141_)), bitcast<f32>(select(_e1144_, 0u, _e1149_)), bitcast<f32>(select(_e1152_, 0u, _e1157_)));
                            }
                            let _e1164_ = (_e939_ + 5u);
                            let _e1169_ = bitcast<u32>(_e903_.member_2_.member_1_.member[0]);
                            let _e1170_ = (_e1169_ & 2139095040u);
                            if (_e1170_ == 2139095040u) {
                                phi_2127_ = true;
                            } else {
                                phi_2127_ = (_e1170_ == 0u);
                            }
                            let _e1174_ = phi_2127_;
                            let _e1177_ = bitcast<u32>(_e903_.member_2_.member_1_.member[1]);
                            let _e1178_ = (_e1177_ & 2139095040u);
                            if (_e1178_ == 2139095040u) {
                                phi_2143_ = true;
                            } else {
                                phi_2143_ = (_e1178_ == 0u);
                            }
                            let _e1182_ = phi_2143_;
                            let _e1185_ = bitcast<u32>(_e903_.member_2_.member_1_.member[2]);
                            let _e1186_ = (_e1185_ & 2139095040u);
                            if (_e1186_ == 2139095040u) {
                                phi_2159_ = true;
                            } else {
                                phi_2159_ = (_e1186_ == 0u);
                            }
                            let _e1190_ = phi_2159_;
                            let _e1193_ = bitcast<u32>(_e903_.member_2_.member_1_.member[3]);
                            let _e1194_ = (_e1193_ & 2139095040u);
                            if (_e1194_ == 2139095040u) {
                                phi_2175_ = true;
                            } else {
                                phi_2175_ = (_e1194_ == 0u);
                            }
                            let _e1198_ = phi_2175_;
                            let _e1202_ = (_e1164_ < _e111_);
                            if _e1202_ {
                                if _e1202_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1164_] = vec4<f32>(bitcast<f32>(select(_e1169_, 0u, _e1174_)), bitcast<f32>(select(_e1177_, 0u, _e1182_)), bitcast<f32>(select(_e1185_, 0u, _e1190_)), bitcast<f32>(select(_e1193_, 0u, _e1198_)));
                            }
                            let _e1205_ = (_e939_ + 6u);
                            let _e1210_ = bitcast<u32>(_e903_.member_3_.member.member[0]);
                            let _e1211_ = (_e1210_ & 2139095040u);
                            if (_e1211_ == 2139095040u) {
                                phi_2207_ = true;
                            } else {
                                phi_2207_ = (_e1211_ == 0u);
                            }
                            let _e1215_ = phi_2207_;
                            let _e1218_ = bitcast<u32>(_e903_.member_3_.member.member[1]);
                            let _e1219_ = (_e1218_ & 2139095040u);
                            if (_e1219_ == 2139095040u) {
                                phi_2223_ = true;
                            } else {
                                phi_2223_ = (_e1219_ == 0u);
                            }
                            let _e1223_ = phi_2223_;
                            let _e1226_ = bitcast<u32>(_e903_.member_3_.member.member[2]);
                            let _e1227_ = (_e1226_ & 2139095040u);
                            if (_e1227_ == 2139095040u) {
                                phi_2239_ = true;
                            } else {
                                phi_2239_ = (_e1227_ == 0u);
                            }
                            let _e1231_ = phi_2239_;
                            let _e1234_ = bitcast<u32>(_e903_.member_3_.member.member[3]);
                            let _e1235_ = (_e1234_ & 2139095040u);
                            if (_e1235_ == 2139095040u) {
                                phi_2255_ = true;
                            } else {
                                phi_2255_ = (_e1235_ == 0u);
                            }
                            let _e1239_ = phi_2255_;
                            let _e1243_ = (_e1205_ < _e111_);
                            if _e1243_ {
                                if _e1243_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1205_] = vec4<f32>(bitcast<f32>(select(_e1210_, 0u, _e1215_)), bitcast<f32>(select(_e1218_, 0u, _e1223_)), bitcast<f32>(select(_e1226_, 0u, _e1231_)), bitcast<f32>(select(_e1234_, 0u, _e1239_)));
                            }
                            let _e1246_ = (_e939_ + 7u);
                            let _e1251_ = bitcast<u32>(_e903_.member_3_.member_1_.member[0]);
                            let _e1252_ = (_e1251_ & 2139095040u);
                            if (_e1252_ == 2139095040u) {
                                phi_2287_ = true;
                            } else {
                                phi_2287_ = (_e1252_ == 0u);
                            }
                            let _e1256_ = phi_2287_;
                            let _e1259_ = bitcast<u32>(_e903_.member_3_.member_1_.member[1]);
                            let _e1260_ = (_e1259_ & 2139095040u);
                            if (_e1260_ == 2139095040u) {
                                phi_2303_ = true;
                            } else {
                                phi_2303_ = (_e1260_ == 0u);
                            }
                            let _e1264_ = phi_2303_;
                            let _e1267_ = bitcast<u32>(_e903_.member_3_.member_1_.member[2]);
                            let _e1268_ = (_e1267_ & 2139095040u);
                            if (_e1268_ == 2139095040u) {
                                phi_2319_ = true;
                            } else {
                                phi_2319_ = (_e1268_ == 0u);
                            }
                            let _e1272_ = phi_2319_;
                            let _e1275_ = bitcast<u32>(_e903_.member_3_.member_1_.member[3]);
                            let _e1276_ = (_e1275_ & 2139095040u);
                            if (_e1276_ == 2139095040u) {
                                phi_2335_ = true;
                            } else {
                                phi_2335_ = (_e1276_ == 0u);
                            }
                            let _e1280_ = phi_2335_;
                            let _e1284_ = (_e1246_ < _e111_);
                            if _e1284_ {
                                if _e1284_ {
                                } else {
                                    phi_17053_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1246_] = vec4<f32>(bitcast<f32>(select(_e1251_, 0u, _e1256_)), bitcast<f32>(select(_e1259_, 0u, _e1264_)), bitcast<f32>(select(_e1267_, 0u, _e1272_)), bitcast<f32>(select(_e1275_, 0u, _e1280_)));
                            }
                        }
                    }
                    let _e1287_ = (_e114_ < 512u);
                    if _e1287_ {
                    } else {
                        phi_17053_ = true;
                        break;
                    }
                    lane_buf[_e114_] = vec4<f32>(0f, 0f, 0f, 0f);
                    let _e1289_ = (256u + _e114_);
                    let _e1290_ = (_e1289_ < 512u);
                    if _e1290_ {
                    } else {
                        phi_17053_ = true;
                        break;
                    }
                    lane_buf[_e1289_] = vec4<f32>(0f, 0f, 0f, 0f);
                    let _e1293_ = ((_e125_ * _e128_) + (_e923_ + _e114_));
                    let _e1294_ = (_e1293_ * 2u);
                    if _e931_ {
                        let _e1295_ = (_e1294_ < _e107_);
                        if _e1295_ {
                            if _e1295_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            let _e1298_ = core_inputs.member[_e1294_];
                            phi_2382_ = _e1298_;
                        } else {
                            phi_2382_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e1300_ = phi_2382_;
                        let _e1301_ = (_e1294_ + 1u);
                        let _e1302_ = (_e1301_ < _e107_);
                        if _e1302_ {
                            if _e1302_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            let _e1305_ = core_inputs.member[_e1301_];
                            phi_2395_ = _e1305_;
                        } else {
                            phi_2395_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e1307_ = phi_2395_;
                        phi_2396_ = _e1307_;
                        phi_2397_ = _e1300_;
                    } else {
                        phi_2396_ = vec4<f32>(0f, 0f, 0f, 0f);
                        phi_2397_ = vec4<f32>(0f, 0f, 0f, 0f);
                    }
                    let _e1309_ = phi_2396_;
                    let _e1311_ = phi_2397_;
                    local = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_1_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_2_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_3_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_4_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_5_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    phi_16474_ = _e899_;
                    phi_2414_ = type_16_(0u, 4u);
                    phi_2417_ = _e903_;
                    loop {
                        let _e1323_ = phi_16474_;
                        let _e1325_ = phi_2414_;
                        let _e1327_ = phi_2417_;
                        local_15_ = _e1327_;
                        local_16_ = _e1327_;
                        if (_e1325_.member < _e1325_.member_1_) {
                            phi_2415_ = type_16_((_e1325_.member + 1u), _e1325_.member_1_);
                            phi_2440_ = type_16_(1u, _e1325_.member);
                        } else {
                            phi_2415_ = _e1325_;
                            phi_2440_ = type_16_(0u, u32());
                        }
                        let _e1340_ = phi_2415_;
                        let _e1342_ = phi_2440_;
                        let _e1346_ = (bitcast<i32>(_e1342_.member) != 0i);
                        if _e1346_ {
                            let _e1347_ = (_e1342_.member_1_ & 3u);
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2458_ = _e294_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2458_ = _e294_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2458_ = _e294_.z;
                                    break;
                                }
                                default: {
                                    phi_2458_ = _e294_.w;
                                    break;
                                }
                            }
                            let _e1354_ = phi_2458_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2469_ = _e301_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2469_ = _e301_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2469_ = _e301_.z;
                                    break;
                                }
                                default: {
                                    phi_2469_ = _e301_.w;
                                    break;
                                }
                            }
                            let _e1361_ = phi_2469_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2480_ = _e308_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2480_ = _e308_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2480_ = _e308_.z;
                                    break;
                                }
                                default: {
                                    phi_2480_ = _e308_.w;
                                    break;
                                }
                            }
                            let _e1368_ = phi_2480_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2491_ = _e315_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2491_ = _e315_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2491_ = _e315_.z;
                                    break;
                                }
                                default: {
                                    phi_2491_ = _e315_.w;
                                    break;
                                }
                            }
                            let _e1375_ = phi_2491_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2502_ = _e322_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2502_ = _e322_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2502_ = _e322_.z;
                                    break;
                                }
                                default: {
                                    phi_2502_ = _e322_.w;
                                    break;
                                }
                            }
                            let _e1382_ = phi_2502_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2513_ = _e329_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2513_ = _e329_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2513_ = _e329_.z;
                                    break;
                                }
                                default: {
                                    phi_2513_ = _e329_.w;
                                    break;
                                }
                            }
                            let _e1389_ = phi_2513_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2535_ = _e224_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2535_ = _e224_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2535_ = _e224_.z;
                                    break;
                                }
                                default: {
                                    phi_2535_ = _e224_.w;
                                    break;
                                }
                            }
                            let _e1396_ = phi_2535_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2546_ = _e231_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2546_ = _e231_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2546_ = _e231_.z;
                                    break;
                                }
                                default: {
                                    phi_2546_ = _e231_.w;
                                    break;
                                }
                            }
                            let _e1403_ = phi_2546_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2557_ = _e238_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2557_ = _e238_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2557_ = _e238_.z;
                                    break;
                                }
                                default: {
                                    phi_2557_ = _e238_.w;
                                    break;
                                }
                            }
                            let _e1410_ = phi_2557_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2568_ = _e245_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2568_ = _e245_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2568_ = _e245_.z;
                                    break;
                                }
                                default: {
                                    phi_2568_ = _e245_.w;
                                    break;
                                }
                            }
                            let _e1417_ = phi_2568_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2579_ = _e252_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2579_ = _e252_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2579_ = _e252_.z;
                                    break;
                                }
                                default: {
                                    phi_2579_ = _e252_.w;
                                    break;
                                }
                            }
                            let _e1424_ = phi_2579_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2590_ = _e259_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2590_ = _e259_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2590_ = _e259_.z;
                                    break;
                                }
                                default: {
                                    phi_2590_ = _e259_.w;
                                    break;
                                }
                            }
                            let _e1431_ = phi_2590_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2601_ = _e266_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2601_ = _e266_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2601_ = _e266_.z;
                                    break;
                                }
                                default: {
                                    phi_2601_ = _e266_.w;
                                    break;
                                }
                            }
                            let _e1438_ = phi_2601_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2612_ = _e273_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2612_ = _e273_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2612_ = _e273_.z;
                                    break;
                                }
                                default: {
                                    phi_2612_ = _e273_.w;
                                    break;
                                }
                            }
                            let _e1445_ = phi_2612_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2623_ = _e280_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2623_ = _e280_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2623_ = _e280_.z;
                                    break;
                                }
                                default: {
                                    phi_2623_ = _e280_.w;
                                    break;
                                }
                            }
                            let _e1452_ = phi_2623_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2634_ = _e287_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2634_ = _e287_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2634_ = _e287_.z;
                                    break;
                                }
                                default: {
                                    phi_2634_ = _e287_.w;
                                    break;
                                }
                            }
                            let _e1459_ = phi_2634_;
                            let _e1460_ = bitcast<u32>(_e1354_);
                            let _e1461_ = (_e1460_ & 2139095040u);
                            if (_e1461_ != 2139095040u) {
                                if (_e1354_ <= 0f) {
                                    phi_2666_ = 0f;
                                } else {
                                    if (_e1354_ >= 0.999995f) {
                                        phi_2665_ = 0.999995f;
                                    } else {
                                        if (_e1461_ == 2139095040u) {
                                            phi_2659_ = true;
                                        } else {
                                            phi_2659_ = (_e1461_ == 0u);
                                        }
                                        let _e1470_ = phi_2659_;
                                        phi_2665_ = bitcast<f32>(select(_e1460_, 0u, _e1470_));
                                    }
                                    let _e1474_ = phi_2665_;
                                    phi_2666_ = _e1474_;
                                }
                                let _e1476_ = phi_2666_;
                                phi_2673_ = _e1476_;
                            } else {
                                phi_2673_ = select(0f, 0.999995f, (_e1460_ == 2139095040u));
                            }
                            let _e1478_ = phi_2673_;
                            let _e1479_ = bitcast<u32>(_e1361_);
                            let _e1480_ = (_e1479_ & 2139095040u);
                            if (_e1480_ != 2139095040u) {
                                if (_e1361_ <= 0f) {
                                    phi_2705_ = 0f;
                                } else {
                                    if (_e1361_ >= 1f) {
                                        phi_2704_ = 1f;
                                    } else {
                                        if (_e1480_ == 2139095040u) {
                                            phi_2698_ = true;
                                        } else {
                                            phi_2698_ = (_e1480_ == 0u);
                                        }
                                        let _e1489_ = phi_2698_;
                                        phi_2704_ = bitcast<f32>(select(_e1479_, 0u, _e1489_));
                                    }
                                    let _e1493_ = phi_2704_;
                                    phi_2705_ = _e1493_;
                                }
                                let _e1495_ = phi_2705_;
                                phi_2712_ = _e1495_;
                            } else {
                                phi_2712_ = select(0f, 1f, (_e1479_ == 2139095040u));
                            }
                            let _e1497_ = phi_2712_;
                            let _e1498_ = bitcast<u32>(_e1368_);
                            let _e1499_ = (_e1498_ & 2139095040u);
                            if (_e1499_ != 2139095040u) {
                                if (_e1368_ <= 0f) {
                                    phi_2744_ = 0f;
                                } else {
                                    if (_e1368_ >= 1f) {
                                        phi_2743_ = 1f;
                                    } else {
                                        if (_e1499_ == 2139095040u) {
                                            phi_2737_ = true;
                                        } else {
                                            phi_2737_ = (_e1499_ == 0u);
                                        }
                                        let _e1508_ = phi_2737_;
                                        phi_2743_ = bitcast<f32>(select(_e1498_, 0u, _e1508_));
                                    }
                                    let _e1512_ = phi_2743_;
                                    phi_2744_ = _e1512_;
                                }
                                let _e1514_ = phi_2744_;
                                phi_2751_ = _e1514_;
                            } else {
                                phi_2751_ = select(0f, 1f, (_e1498_ == 2139095040u));
                            }
                            let _e1516_ = phi_2751_;
                            let _e1517_ = bitcast<u32>(_e1375_);
                            let _e1518_ = (_e1517_ & 2139095040u);
                            if (_e1518_ != 2139095040u) {
                                if (_e1375_ <= 0f) {
                                    phi_2783_ = 0f;
                                } else {
                                    if (_e1375_ >= 64f) {
                                        phi_2782_ = 64f;
                                    } else {
                                        if (_e1518_ == 2139095040u) {
                                            phi_2776_ = true;
                                        } else {
                                            phi_2776_ = (_e1518_ == 0u);
                                        }
                                        let _e1527_ = phi_2776_;
                                        phi_2782_ = bitcast<f32>(select(_e1517_, 0u, _e1527_));
                                    }
                                    let _e1531_ = phi_2782_;
                                    phi_2783_ = _e1531_;
                                }
                                let _e1533_ = phi_2783_;
                                phi_2790_ = _e1533_;
                            } else {
                                phi_2790_ = select(0f, 64f, (_e1517_ == 2139095040u));
                            }
                            let _e1535_ = phi_2790_;
                            let _e1536_ = bitcast<u32>(_e1382_);
                            let _e1537_ = (_e1536_ & 2139095040u);
                            if (_e1537_ != 2139095040u) {
                                if (_e1382_ <= 0f) {
                                    phi_2822_ = 0f;
                                } else {
                                    if (_e1382_ >= 1f) {
                                        phi_2821_ = 1f;
                                    } else {
                                        if (_e1537_ == 2139095040u) {
                                            phi_2815_ = true;
                                        } else {
                                            phi_2815_ = (_e1537_ == 0u);
                                        }
                                        let _e1546_ = phi_2815_;
                                        phi_2821_ = bitcast<f32>(select(_e1536_, 0u, _e1546_));
                                    }
                                    let _e1550_ = phi_2821_;
                                    phi_2822_ = _e1550_;
                                }
                                let _e1552_ = phi_2822_;
                                phi_2829_ = _e1552_;
                            } else {
                                phi_2829_ = select(0f, 1f, (_e1536_ == 2139095040u));
                            }
                            let _e1554_ = phi_2829_;
                            let _e1555_ = bitcast<u32>(_e1389_);
                            let _e1556_ = (_e1555_ & 2139095040u);
                            if (_e1556_ != 2139095040u) {
                                if (_e1389_ <= 0f) {
                                    phi_2861_ = 0f;
                                } else {
                                    if (_e1389_ >= 1f) {
                                        phi_2860_ = 1f;
                                    } else {
                                        if (_e1556_ == 2139095040u) {
                                            phi_2854_ = true;
                                        } else {
                                            phi_2854_ = (_e1556_ == 0u);
                                        }
                                        let _e1565_ = phi_2854_;
                                        phi_2860_ = bitcast<f32>(select(_e1555_, 0u, _e1565_));
                                    }
                                    let _e1569_ = phi_2860_;
                                    phi_2861_ = _e1569_;
                                }
                                let _e1571_ = phi_2861_;
                                phi_2868_ = _e1571_;
                            } else {
                                phi_2868_ = select(0f, 1f, (_e1555_ == 2139095040u));
                            }
                            let _e1573_ = phi_2868_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2939_ = _e336_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2939_ = _e336_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2939_ = _e336_.z;
                                    break;
                                }
                                default: {
                                    phi_2939_ = _e336_.w;
                                    break;
                                }
                            }
                            let _e1580_ = phi_2939_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2950_ = _e343_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2950_ = _e343_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2950_ = _e343_.z;
                                    break;
                                }
                                default: {
                                    phi_2950_ = _e343_.w;
                                    break;
                                }
                            }
                            let _e1587_ = phi_2950_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2961_ = _e350_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2961_ = _e350_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2961_ = _e350_.z;
                                    break;
                                }
                                default: {
                                    phi_2961_ = _e350_.w;
                                    break;
                                }
                            }
                            let _e1594_ = phi_2961_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2972_ = _e357_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2972_ = _e357_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2972_ = _e357_.z;
                                    break;
                                }
                                default: {
                                    phi_2972_ = _e357_.w;
                                    break;
                                }
                            }
                            let _e1601_ = phi_2972_;
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_2983_ = _e364_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2983_ = _e364_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2983_ = _e364_.z;
                                    break;
                                }
                                default: {
                                    phi_2983_ = _e364_.w;
                                    break;
                                }
                            }
                            let _e1608_ = phi_2983_;
                            if _e931_ {
                                switch bitcast<i32>(_e1347_) {
                                    case 0: {
                                        phi_3047_ = _e1311_.x;
                                        break;
                                    }
                                    case 1: {
                                        phi_3047_ = _e1311_.y;
                                        break;
                                    }
                                    case 2: {
                                        phi_3047_ = _e1311_.z;
                                        break;
                                    }
                                    default: {
                                        phi_3047_ = _e1311_.w;
                                        break;
                                    }
                                }
                                let _e1615_ = phi_3047_;
                                let _e1616_ = bitcast<u32>(_e1615_);
                                let _e1617_ = (_e1616_ & 2139095040u);
                                if (_e1617_ == 2139095040u) {
                                    phi_3058_ = true;
                                } else {
                                    phi_3058_ = (_e1617_ == 0u);
                                }
                                let _e1621_ = phi_3058_;
                                phi_3064_ = bitcast<f32>(select(_e1616_, 0u, _e1621_));
                            } else {
                                phi_3064_ = 0f;
                            }
                            let _e1625_ = phi_3064_;
                            if _e931_ {
                                switch bitcast<i32>(_e1347_) {
                                    case 0: {
                                        phi_3078_ = _e1309_.x;
                                        break;
                                    }
                                    case 1: {
                                        phi_3078_ = _e1309_.y;
                                        break;
                                    }
                                    case 2: {
                                        phi_3078_ = _e1309_.z;
                                        break;
                                    }
                                    default: {
                                        phi_3078_ = _e1309_.w;
                                        break;
                                    }
                                }
                                let _e1632_ = phi_3078_;
                                let _e1633_ = bitcast<u32>(_e1632_);
                                let _e1634_ = (_e1633_ & 2139095040u);
                                if (_e1634_ == 2139095040u) {
                                    phi_3089_ = true;
                                } else {
                                    phi_3089_ = (_e1634_ == 0u);
                                }
                                let _e1638_ = phi_3089_;
                                phi_3095_ = bitcast<f32>(select(_e1633_, 0u, _e1638_));
                            } else {
                                phi_3095_ = 0f;
                            }
                            let _e1642_ = phi_3095_;
                            if _e931_ {
                                let _e1643_ = (_e1403_ + _e1403_);
                                let _e1644_ = (_e1410_ + _e1410_);
                                phi_3109_ = type_25_(((_e1396_ + _e1396_) - 1f), -(_e1643_), _e1643_, (1f - _e1644_), (_e1643_ * _e1625_), (_e1644_ * _e1625_));
                            } else {
                                phi_3109_ = type_25_(1f, 0f, 0f, 1f, 0f, 0f);
                            }
                            let _e1653_ = phi_3109_;
                            local_6_ = array<f32, 4>(_e903_.member.member.member[0], _e903_.member.member.member[1], _e903_.member.member.member[2], _e903_.member.member.member[3]);
                            let _e1659_ = (_e1347_ < 4u);
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e1661_ = local_6_[_e1347_];
                            local_7_ = array<f32, 4>(_e903_.member_1_.member.member[0], _e903_.member_1_.member.member[1], _e903_.member_1_.member.member[2], _e903_.member_1_.member.member[3]);
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e1668_ = local_7_[_e1347_];
                            let _e1669_ = array<f32, 2>(_e1661_, _e1668_);
                            let _e1670_ = (_e114_ & 255u);
                            let _e1671_ = (_e1670_ & 255u);
                            let _e1673_ = (_e1671_ < 1536u);
                            if _e1673_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1671_] = _e1653_.member;
                            let _e1676_ = (256u + _e1671_);
                            let _e1677_ = (_e1676_ < 1536u);
                            if _e1677_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1676_] = _e1653_.member_1_;
                            let _e1680_ = (512u + _e1671_);
                            let _e1681_ = (_e1680_ < 1536u);
                            if _e1681_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1680_] = _e1653_.member_2_;
                            let _e1684_ = (768u + _e1671_);
                            let _e1685_ = (_e1684_ < 1536u);
                            if _e1685_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1684_] = _e1653_.member_3_;
                            let _e1688_ = (1024u + _e1671_);
                            let _e1689_ = (_e1688_ < 1536u);
                            if _e1689_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1688_] = _e1653_.member_4_;
                            let _e1692_ = (1280u + _e1671_);
                            let _e1693_ = (_e1692_ < 1536u);
                            if _e1693_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1692_] = _e1653_.member_5_;
                            workgroupBarrier();
                            phi_3187_ = 1u;
                            phi_3190_ = _e1653_;
                            loop {
                                let _e1696_ = phi_3187_;
                                let _e1698_ = phi_3190_;
                                let _e1699_ = (_e1696_ < 256u);
                                if _e1699_ {
                                    if (_e1670_ >= _e1696_) {
                                        let _e1702_ = ((_e1670_ - _e1696_) & 255u);
                                        if (_e1702_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1705_ = scan_scratch[_e1702_];
                                        let _e1706_ = (256u + _e1702_);
                                        if (_e1706_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1709_ = scan_scratch[_e1706_];
                                        let _e1710_ = (512u + _e1702_);
                                        if (_e1710_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1713_ = scan_scratch[_e1710_];
                                        let _e1714_ = (768u + _e1702_);
                                        if (_e1714_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1717_ = scan_scratch[_e1714_];
                                        let _e1718_ = (1024u + _e1702_);
                                        if (_e1718_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1721_ = scan_scratch[_e1718_];
                                        let _e1722_ = (1280u + _e1702_);
                                        if (_e1722_ < 1536u) {
                                        } else {
                                            phi_16434_ = true;
                                            break;
                                        }
                                        let _e1725_ = scan_scratch[_e1722_];
                                        phi_3286_ = type_25_(((_e1698_.member * _e1705_) + (_e1698_.member_1_ * _e1713_)), ((_e1698_.member * _e1709_) + (_e1698_.member_1_ * _e1717_)), ((_e1698_.member_2_ * _e1705_) + (_e1698_.member_3_ * _e1713_)), ((_e1698_.member_2_ * _e1709_) + (_e1698_.member_3_ * _e1717_)), (((_e1698_.member * _e1721_) + (_e1698_.member_1_ * _e1725_)) + _e1698_.member_4_), (((_e1698_.member_2_ * _e1721_) + (_e1698_.member_3_ * _e1725_)) + _e1698_.member_5_));
                                    } else {
                                        phi_3286_ = _e1698_;
                                    }
                                    let _e1754_ = phi_3286_;
                                    workgroupBarrier();
                                    if _e1673_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1671_] = _e1754_.member;
                                    if _e1677_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1676_] = _e1754_.member_1_;
                                    if _e1681_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1680_] = _e1754_.member_2_;
                                    if _e1685_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1684_] = _e1754_.member_3_;
                                    if _e1689_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1688_] = _e1754_.member_4_;
                                    if _e1693_ {
                                    } else {
                                        phi_16434_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1692_] = _e1754_.member_5_;
                                    workgroupBarrier();
                                    phi_3188_ = (_e1696_ << bitcast<u32>(1i));
                                    phi_3191_ = _e1754_;
                                } else {
                                    phi_3188_ = u32();
                                    phi_3191_ = type_25_();
                                }
                                let _e1764_ = phi_3188_;
                                let _e1766_ = phi_3191_;
                                phi_3187_ = _e1764_;
                                phi_3190_ = _e1766_;
                                phi_16434_ = _e1323_;
                                if !(_e1699_) {
                                    break;
                                }
                            }
                            let _e1769_ = phi_16434_;
                            phi_17036_ = _e1769_;
                            if _e1769_ {
                                break;
                            }
                            let _e1770_ = (_e1670_ == 0u);
                            if _e1770_ {
                                phi_3398_ = _e1669_;
                            } else {
                                let _e1772_ = ((_e1670_ - 1u) & 255u);
                                if (_e1772_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1775_ = scan_scratch[_e1772_];
                                let _e1776_ = (256u + _e1772_);
                                if (_e1776_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1779_ = scan_scratch[_e1776_];
                                let _e1780_ = (512u + _e1772_);
                                if (_e1780_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1783_ = scan_scratch[_e1780_];
                                let _e1784_ = (768u + _e1772_);
                                if (_e1784_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1787_ = scan_scratch[_e1784_];
                                let _e1788_ = (1024u + _e1772_);
                                if (_e1788_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1791_ = scan_scratch[_e1788_];
                                let _e1792_ = (1280u + _e1772_);
                                if (_e1792_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1795_ = scan_scratch[_e1792_];
                                phi_3398_ = array<f32, 2>((((_e1775_ * _e1661_) + (_e1779_ * _e1668_)) + _e1791_), (((_e1783_ * _e1661_) + (_e1787_ * _e1668_)) + _e1795_));
                            }
                            let _e1806_ = phi_3398_;
                            let _e1809_ = (_e929_ > 0u);
                            if _e1809_ {
                                if (_e929_ <= 256u) {
                                    phi_3410_ = (_e929_ - 1u);
                                } else {
                                    phi_3410_ = 255u;
                                }
                                let _e1813_ = phi_3410_;
                                let _e1814_ = (_e1813_ & 255u);
                                if (_e1814_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1817_ = scan_scratch[_e1814_];
                                let _e1818_ = (256u + _e1814_);
                                if (_e1818_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1821_ = scan_scratch[_e1818_];
                                let _e1822_ = (512u + _e1814_);
                                if (_e1822_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1825_ = scan_scratch[_e1822_];
                                let _e1826_ = (768u + _e1814_);
                                if (_e1826_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1829_ = scan_scratch[_e1826_];
                                let _e1830_ = (1024u + _e1814_);
                                if (_e1830_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1833_ = scan_scratch[_e1830_];
                                let _e1834_ = (1280u + _e1814_);
                                if (_e1834_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1837_ = scan_scratch[_e1834_];
                                phi_3468_ = array<f32, 2>((((_e1817_ * _e1661_) + (_e1821_ * _e1668_)) + _e1833_), (((_e1825_ * _e1661_) + (_e1829_ * _e1668_)) + _e1837_));
                            } else {
                                phi_3468_ = _e1669_;
                            }
                            let _e1848_ = phi_3468_;
                            workgroupBarrier();
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local_2_[_e1347_] = _e1848_[0];
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local_4_[_e1347_] = _e1848_[1];
                            let _e1853_ = (_e1625_ - _e1806_[1]);
                            let _e1856_ = ((_e1396_ * _e1806_[0]) + (_e1403_ * _e1853_));
                            let _e1860_ = ((_e1806_[1] + (_e1403_ * _e1806_[0])) + (_e1410_ * _e1853_));
                            let _e1875_ = (_e114_ < 256u);
                            if _e1875_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            aux_scratch[_e114_] = (((((_e1860_ * _e1431_) + (_e1856_ * _e1438_)) + (((_e1625_ - (_e1417_ * _e1856_)) - _e1860_) * _e1445_)) + ((_e1625_ - (_e1424_ * _e1856_)) * _e1452_)) + (_e1625_ * _e1459_));
                            if _e1287_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    lane_buf[_e114_].x = _e1856_;
                                    break;
                                }
                                case 1: {
                                    lane_buf[_e114_].y = _e1856_;
                                    break;
                                }
                                case 2: {
                                    lane_buf[_e114_].z = _e1856_;
                                    break;
                                }
                                default: {
                                    lane_buf[_e114_].w = _e1856_;
                                    break;
                                }
                            }
                            if _e931_ {
                                let _e1886_ = (_e1403_ + _e1403_);
                                let _e1887_ = (_e1410_ + _e1410_);
                                phi_3729_ = type_25_(((_e1396_ + _e1396_) - 1f), -(_e1886_), _e1886_, (1f - _e1887_), (_e1886_ * _e1642_), (_e1887_ * _e1642_));
                            } else {
                                phi_3729_ = type_25_(1f, 0f, 0f, 1f, 0f, 0f);
                            }
                            let _e1896_ = phi_3729_;
                            local_8_ = array<f32, 4>(_e903_.member.member_1_.member[0], _e903_.member.member_1_.member[1], _e903_.member.member_1_.member[2], _e903_.member.member_1_.member[3]);
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e1903_ = local_8_[_e1347_];
                            local_9_ = array<f32, 4>(_e903_.member_1_.member_1_.member[0], _e903_.member_1_.member_1_.member[1], _e903_.member_1_.member_1_.member[2], _e903_.member_1_.member_1_.member[3]);
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e1910_ = local_9_[_e1347_];
                            let _e1911_ = array<f32, 2>(_e1903_, _e1910_);
                            if _e1673_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1671_] = _e1896_.member;
                            if _e1677_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1676_] = _e1896_.member_1_;
                            if _e1681_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1680_] = _e1896_.member_2_;
                            if _e1685_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1684_] = _e1896_.member_3_;
                            if _e1689_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1688_] = _e1896_.member_4_;
                            if _e1693_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1692_] = _e1896_.member_5_;
                            workgroupBarrier();
                            phi_3807_ = 1u;
                            phi_3810_ = _e1896_;
                            loop {
                                let _e1919_ = phi_3807_;
                                let _e1921_ = phi_3810_;
                                let _e1922_ = (_e1919_ < 256u);
                                if _e1922_ {
                                    if (_e1670_ >= _e1919_) {
                                        let _e1925_ = ((_e1670_ - _e1919_) & 255u);
                                        if (_e1925_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1928_ = scan_scratch[_e1925_];
                                        let _e1929_ = (256u + _e1925_);
                                        if (_e1929_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1932_ = scan_scratch[_e1929_];
                                        let _e1933_ = (512u + _e1925_);
                                        if (_e1933_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1936_ = scan_scratch[_e1933_];
                                        let _e1937_ = (768u + _e1925_);
                                        if (_e1937_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1940_ = scan_scratch[_e1937_];
                                        let _e1941_ = (1024u + _e1925_);
                                        if (_e1941_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1944_ = scan_scratch[_e1941_];
                                        let _e1945_ = (1280u + _e1925_);
                                        if (_e1945_ < 1536u) {
                                        } else {
                                            phi_16747_ = true;
                                            break;
                                        }
                                        let _e1948_ = scan_scratch[_e1945_];
                                        phi_3906_ = type_25_(((_e1921_.member * _e1928_) + (_e1921_.member_1_ * _e1936_)), ((_e1921_.member * _e1932_) + (_e1921_.member_1_ * _e1940_)), ((_e1921_.member_2_ * _e1928_) + (_e1921_.member_3_ * _e1936_)), ((_e1921_.member_2_ * _e1932_) + (_e1921_.member_3_ * _e1940_)), (((_e1921_.member * _e1944_) + (_e1921_.member_1_ * _e1948_)) + _e1921_.member_4_), (((_e1921_.member_2_ * _e1944_) + (_e1921_.member_3_ * _e1948_)) + _e1921_.member_5_));
                                    } else {
                                        phi_3906_ = _e1921_;
                                    }
                                    let _e1977_ = phi_3906_;
                                    workgroupBarrier();
                                    if _e1673_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1671_] = _e1977_.member;
                                    if _e1677_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1676_] = _e1977_.member_1_;
                                    if _e1681_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1680_] = _e1977_.member_2_;
                                    if _e1685_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1684_] = _e1977_.member_3_;
                                    if _e1689_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1688_] = _e1977_.member_4_;
                                    if _e1693_ {
                                    } else {
                                        phi_16747_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1692_] = _e1977_.member_5_;
                                    workgroupBarrier();
                                    phi_3808_ = (_e1919_ << bitcast<u32>(1i));
                                    phi_3811_ = _e1977_;
                                } else {
                                    phi_3808_ = u32();
                                    phi_3811_ = type_25_();
                                }
                                let _e1987_ = phi_3808_;
                                let _e1989_ = phi_3811_;
                                phi_3807_ = _e1987_;
                                phi_3810_ = _e1989_;
                                phi_16747_ = _e1769_;
                                if !(_e1922_) {
                                    break;
                                }
                            }
                            let _e1992_ = phi_16747_;
                            phi_17036_ = _e1992_;
                            if _e1992_ {
                                break;
                            }
                            if _e1770_ {
                                phi_4018_ = _e1911_;
                            } else {
                                let _e1994_ = ((_e1670_ - 1u) & 255u);
                                if (_e1994_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e1997_ = scan_scratch[_e1994_];
                                let _e1998_ = (256u + _e1994_);
                                if (_e1998_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2001_ = scan_scratch[_e1998_];
                                let _e2002_ = (512u + _e1994_);
                                if (_e2002_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2005_ = scan_scratch[_e2002_];
                                let _e2006_ = (768u + _e1994_);
                                if (_e2006_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2009_ = scan_scratch[_e2006_];
                                let _e2010_ = (1024u + _e1994_);
                                if (_e2010_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2013_ = scan_scratch[_e2010_];
                                let _e2014_ = (1280u + _e1994_);
                                if (_e2014_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2017_ = scan_scratch[_e2014_];
                                phi_4018_ = array<f32, 2>((((_e1997_ * _e1903_) + (_e2001_ * _e1910_)) + _e2013_), (((_e2005_ * _e1903_) + (_e2009_ * _e1910_)) + _e2017_));
                            }
                            let _e2028_ = phi_4018_;
                            if _e1809_ {
                                if (_e929_ <= 256u) {
                                    phi_4030_ = (_e929_ - 1u);
                                } else {
                                    phi_4030_ = 255u;
                                }
                                let _e2034_ = phi_4030_;
                                let _e2035_ = (_e2034_ & 255u);
                                if (_e2035_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2038_ = scan_scratch[_e2035_];
                                let _e2039_ = (256u + _e2035_);
                                if (_e2039_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2042_ = scan_scratch[_e2039_];
                                let _e2043_ = (512u + _e2035_);
                                if (_e2043_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2046_ = scan_scratch[_e2043_];
                                let _e2047_ = (768u + _e2035_);
                                if (_e2047_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2050_ = scan_scratch[_e2047_];
                                let _e2051_ = (1024u + _e2035_);
                                if (_e2051_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2054_ = scan_scratch[_e2051_];
                                let _e2055_ = (1280u + _e2035_);
                                if (_e2055_ < 1536u) {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2058_ = scan_scratch[_e2055_];
                                phi_4088_ = array<f32, 2>((((_e2038_ * _e1903_) + (_e2042_ * _e1910_)) + _e2054_), (((_e2046_ * _e1903_) + (_e2050_ * _e1910_)) + _e2058_));
                            } else {
                                phi_4088_ = _e1911_;
                            }
                            let _e2069_ = phi_4088_;
                            workgroupBarrier();
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local_3_[_e1347_] = _e2069_[0];
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local_5_[_e1347_] = _e2069_[1];
                            let _e2074_ = (_e1642_ - _e2028_[1]);
                            let _e2077_ = ((_e1396_ * _e2028_[0]) + (_e1403_ * _e2074_));
                            let _e2081_ = ((_e2028_[1] + (_e1403_ * _e2028_[0])) + (_e1410_ * _e2074_));
                            let _e2095_ = (((((_e2081_ * _e1431_) + (_e2077_ * _e1438_)) + (((_e1642_ - (_e1417_ * _e2077_)) - _e2081_) * _e1445_)) + ((_e1642_ - (_e1424_ * _e2077_)) * _e1452_)) + (_e1642_ * _e1459_));
                            if _e1875_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e2096_ = aux_scratch[_e114_];
                            if _e1287_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            let _e2097_ = lane_buf[_e114_];
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    phi_4338_ = _e2097_.x;
                                    break;
                                }
                                case 1: {
                                    phi_4338_ = _e2097_.y;
                                    break;
                                }
                                case 2: {
                                    phi_4338_ = _e2097_.z;
                                    break;
                                }
                                default: {
                                    phi_4338_ = _e2097_.w;
                                    break;
                                }
                            }
                            let _e2104_ = phi_4338_;
                            if (_e114_ < 1536u) {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e114_] = _e2096_;
                            if (_e1289_ < 1536u) {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e1289_] = _e2104_;
                            let _e2109_ = (512u + _e114_);
                            if (_e2109_ < 1536u) {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e2109_] = _e2095_;
                            let _e2112_ = (768u + _e114_);
                            if (_e2112_ < 1536u) {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e2112_] = _e2077_;
                            let _e2115_ = (1024u + _e114_);
                            let _e2116_ = (_e2115_ < 1536u);
                            if _e2116_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e2115_] = 0f;
                            let _e2118_ = (1280u + _e114_);
                            let _e2119_ = (_e2118_ < 1536u);
                            if _e2119_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            scan_scratch[_e2118_] = 0f;
                            workgroupBarrier();
                            if _e932_ {
                                local_10_ = array<f32, 4>(_e1327_.member_2_.member.member[0], _e1327_.member_2_.member.member[1], _e1327_.member_2_.member.member[2], _e1327_.member_2_.member.member[3]);
                                local_11_ = array<f32, 4>(_e1327_.member_2_.member_1_.member[0], _e1327_.member_2_.member_1_.member[1], _e1327_.member_2_.member_1_.member[2], _e1327_.member_2_.member_1_.member[3]);
                                if _e1659_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2137_ = local_10_[_e1347_];
                                if _e1659_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2139_ = local_11_[_e1347_];
                                phi_4431_ = type_16_(0u, _e929_);
                                phi_4434_ = _e2139_;
                                phi_4436_ = _e2137_;
                                loop {
                                    let _e2142_ = phi_4431_;
                                    let _e2144_ = phi_4434_;
                                    let _e2146_ = phi_4436_;
                                    local_13_ = _e2146_;
                                    local_14_ = _e2144_;
                                    if (_e2142_.member < _e2142_.member_1_) {
                                        phi_4432_ = type_16_((_e2142_.member + 1u), _e2142_.member_1_);
                                        phi_4459_ = type_16_(1u, _e2142_.member);
                                    } else {
                                        phi_4432_ = _e2142_;
                                        phi_4459_ = type_16_(0u, u32());
                                    }
                                    let _e2159_ = phi_4432_;
                                    let _e2161_ = phi_4459_;
                                    let _e2165_ = (bitcast<i32>(_e2161_.member) != 0i);
                                    if _e2165_ {
                                        let _e2166_ = (_e2161_.member_1_ & 255u);
                                        if (_e2166_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        let _e2169_ = scan_scratch[_e2166_];
                                        let _e2170_ = (256u + _e2166_);
                                        if (_e2170_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        let _e2173_ = scan_scratch[_e2170_];
                                        let _e2174_ = (512u + _e2166_);
                                        if (_e2174_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        let _e2177_ = scan_scratch[_e2174_];
                                        let _e2178_ = (768u + _e2166_);
                                        if (_e2178_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        let _e2181_ = scan_scratch[_e2178_];
                                        let _e2184_ = bitcast<u32>((_e1580_ * _e2173_));
                                        let _e2185_ = (_e2184_ & 2139095040u);
                                        if (_e2185_ == 2139095040u) {
                                            phi_4513_ = 0u;
                                        } else {
                                            if (_e2185_ == 0u) {
                                                phi_4512_ = (_e2184_ & 2147483648u);
                                            } else {
                                                phi_4512_ = _e2184_;
                                            }
                                            let _e2190_ = phi_4512_;
                                            phi_4513_ = _e2190_;
                                        }
                                        let _e2192_ = phi_4513_;
                                        let _e2198_ = (_e1535_ <= 0f);
                                        let _e2200_ = (1f - _e1497_);
                                        let _e2201_ = (_e2200_ + select((_e1535_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2192_)) & 2147483647u))), 0f, _e2198_));
                                        if (_e2201_ != _e2201_) {
                                            phi_11134_ = true;
                                        } else {
                                            phi_11134_ = (0f >= _e2201_);
                                        }
                                        let _e2205_ = phi_11134_;
                                        let _e2206_ = select(_e2201_, 0f, _e2205_);
                                        if (_e2206_ != _e2206_) {
                                            phi_11194_ = true;
                                        } else {
                                            phi_11194_ = (1f <= _e2206_);
                                        }
                                        let _e2210_ = phi_11194_;
                                        let _e2211_ = select(_e2206_, 1f, _e2210_);
                                        let _e2212_ = (_e1478_ * _e2146_);
                                        let _e2213_ = (1f - _e1554_);
                                        let _e2214_ = (_e1516_ * _e2213_);
                                        let _e2216_ = ((_e1587_ * _e2169_) - (_e2214_ * _e2212_));
                                        if (_e2216_ != _e2216_) {
                                            phi_11254_ = true;
                                        } else {
                                            phi_11254_ = (-1f >= _e2216_);
                                        }
                                        let _e2220_ = phi_11254_;
                                        let _e2221_ = select(_e2216_, -1f, _e2220_);
                                        if (_e2221_ != _e2221_) {
                                            phi_11314_ = true;
                                        } else {
                                            phi_11314_ = (1f <= _e2221_);
                                        }
                                        let _e2225_ = phi_11314_;
                                        let _e2226_ = select(_e2221_, 1f, _e2225_);
                                        let _e2233_ = bitcast<f32>((bitcast<u32>(_e2216_) & 2147483647u));
                                        if (_e2233_ != _e2233_) {
                                            phi_11374_ = true;
                                        } else {
                                            phi_11374_ = (1000000000000000f <= _e2233_);
                                        }
                                        let _e2237_ = phi_11374_;
                                        let _e2238_ = select(_e2233_, 1000000000000000f, _e2237_);
                                        let _e2239_ = (_e2238_ * _e2238_);
                                        let _e2240_ = (_e2239_ + 0.0000010000001f);
                                        if (_e2240_ != _e2240_) {
                                            phi_11434_ = true;
                                        } else {
                                            phi_11434_ = (0.000000000001f >= _e2240_);
                                        }
                                        let _e2244_ = phi_11434_;
                                        let _e2245_ = select(_e2240_, 0.000000000001f, _e2244_);
                                        let _e2246_ = (_e2245_ * 0.5f);
                                        let _e2251_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2245_) >> bitcast<u32>(1i))));
                                        let _e2255_ = (_e2251_ * (1.5f - (_e2246_ * (_e2251_ * _e2251_))));
                                        let _e2259_ = (_e2255_ * (1.5f - (_e2246_ * (_e2255_ * _e2255_))));
                                        let _e2264_ = (_e2240_ * (_e2259_ * (1.5f - (_e2246_ * (_e2259_ * _e2259_)))));
                                        let _e2265_ = (_e2264_ + 0.001f);
                                        if (_e2265_ != _e2265_) {
                                            phi_11494_ = true;
                                        } else {
                                            phi_11494_ = (0.000001f >= _e2265_);
                                        }
                                        let _e2269_ = phi_11494_;
                                        let _e2270_ = select(_e2265_, 0.000001f, _e2269_);
                                        let _e2273_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2270_)));
                                        let _e2277_ = (_e2273_ + (_e2273_ * (1f - (_e2270_ * _e2273_))));
                                        let _e2281_ = (_e2277_ + (_e2277_ * (1f - (_e2270_ * _e2277_))));
                                        let _e2285_ = (_e2281_ + (_e2281_ * (1f - (_e2270_ * _e2281_))));
                                        let _e2286_ = (_e2264_ + _e2238_);
                                        if (_e2286_ != _e2286_) {
                                            phi_11554_ = true;
                                        } else {
                                            phi_11554_ = (0.000001f >= _e2286_);
                                        }
                                        let _e2290_ = phi_11554_;
                                        let _e2291_ = select(_e2286_, 0.000001f, _e2290_);
                                        let _e2294_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2291_)));
                                        let _e2298_ = (_e2294_ + (_e2294_ * (1f - (_e2291_ * _e2294_))));
                                        let _e2302_ = (_e2298_ + (_e2298_ * (1f - (_e2291_ * _e2298_))));
                                        let _e2312_ = (_e2233_ <= 1000000000000000f);
                                        let _e2322_ = select(select(select(_e2233_, (0.5f * (_e2238_ + (_e2239_ * _e2285_))), _e2312_), select(-0.0005f, -(((0.0005f * (_e2238_ * (_e2302_ + (_e2302_ * (1f - (_e2291_ * _e2302_)))))) * (1f + (_e2238_ * _e2285_)))), _e2312_), (_e2216_ <= 0f)), 0f, (_e2233_ <= 0f));
                                        let _e2326_ = (_e1554_ <= 0f);
                                        let _e2333_ = bitcast<u32>((((1f - (_e1554_ * _e2211_)) * _e2212_) + (_e2211_ * select(((_e2213_ * _e2322_) + (_e1554_ * (_e2226_ * (1f - (0.33333334f * (_e2226_ * _e2226_)))))), _e2322_, _e2326_))));
                                        let _e2334_ = (_e2333_ & 2139095040u);
                                        if (_e2334_ == 2139095040u) {
                                            phi_5255_ = 0u;
                                        } else {
                                            if (_e2334_ == 0u) {
                                                phi_5254_ = (_e2333_ & 2147483648u);
                                            } else {
                                                phi_5254_ = _e2333_;
                                            }
                                            let _e2339_ = phi_5254_;
                                            phi_5255_ = _e2339_;
                                        }
                                        let _e2341_ = phi_5255_;
                                        let _e2342_ = bitcast<f32>(_e2341_);
                                        let _e2345_ = bitcast<u32>((_e1580_ * _e2181_));
                                        let _e2346_ = (_e2345_ & 2139095040u);
                                        if (_e2346_ == 2139095040u) {
                                            phi_5317_ = 0u;
                                        } else {
                                            if (_e2346_ == 0u) {
                                                phi_5316_ = (_e2345_ & 2147483648u);
                                            } else {
                                                phi_5316_ = _e2345_;
                                            }
                                            let _e2351_ = phi_5316_;
                                            phi_5317_ = _e2351_;
                                        }
                                        let _e2353_ = phi_5317_;
                                        let _e2360_ = (_e2200_ + select((_e1535_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2353_)) & 2147483647u))), 0f, _e2198_));
                                        if (_e2360_ != _e2360_) {
                                            phi_11734_ = true;
                                        } else {
                                            phi_11734_ = (0f >= _e2360_);
                                        }
                                        let _e2364_ = phi_11734_;
                                        let _e2365_ = select(_e2360_, 0f, _e2364_);
                                        if (_e2365_ != _e2365_) {
                                            phi_11794_ = true;
                                        } else {
                                            phi_11794_ = (1f <= _e2365_);
                                        }
                                        let _e2369_ = phi_11794_;
                                        let _e2370_ = select(_e2365_, 1f, _e2369_);
                                        let _e2371_ = (_e1478_ * _e2144_);
                                        let _e2373_ = ((_e1587_ * _e2177_) - (_e2214_ * _e2371_));
                                        if (_e2373_ != _e2373_) {
                                            phi_11854_ = true;
                                        } else {
                                            phi_11854_ = (-1f >= _e2373_);
                                        }
                                        let _e2377_ = phi_11854_;
                                        let _e2378_ = select(_e2373_, -1f, _e2377_);
                                        if (_e2378_ != _e2378_) {
                                            phi_11914_ = true;
                                        } else {
                                            phi_11914_ = (1f <= _e2378_);
                                        }
                                        let _e2382_ = phi_11914_;
                                        let _e2383_ = select(_e2378_, 1f, _e2382_);
                                        let _e2390_ = bitcast<f32>((bitcast<u32>(_e2373_) & 2147483647u));
                                        if (_e2390_ != _e2390_) {
                                            phi_11974_ = true;
                                        } else {
                                            phi_11974_ = (1000000000000000f <= _e2390_);
                                        }
                                        let _e2394_ = phi_11974_;
                                        let _e2395_ = select(_e2390_, 1000000000000000f, _e2394_);
                                        let _e2396_ = (_e2395_ * _e2395_);
                                        let _e2397_ = (_e2396_ + 0.0000010000001f);
                                        if (_e2397_ != _e2397_) {
                                            phi_12034_ = true;
                                        } else {
                                            phi_12034_ = (0.000000000001f >= _e2397_);
                                        }
                                        let _e2401_ = phi_12034_;
                                        let _e2402_ = select(_e2397_, 0.000000000001f, _e2401_);
                                        let _e2403_ = (_e2402_ * 0.5f);
                                        let _e2408_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2402_) >> bitcast<u32>(1i))));
                                        let _e2412_ = (_e2408_ * (1.5f - (_e2403_ * (_e2408_ * _e2408_))));
                                        let _e2416_ = (_e2412_ * (1.5f - (_e2403_ * (_e2412_ * _e2412_))));
                                        let _e2421_ = (_e2397_ * (_e2416_ * (1.5f - (_e2403_ * (_e2416_ * _e2416_)))));
                                        let _e2422_ = (_e2421_ + 0.001f);
                                        if (_e2422_ != _e2422_) {
                                            phi_12094_ = true;
                                        } else {
                                            phi_12094_ = (0.000001f >= _e2422_);
                                        }
                                        let _e2426_ = phi_12094_;
                                        let _e2427_ = select(_e2422_, 0.000001f, _e2426_);
                                        let _e2430_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2427_)));
                                        let _e2434_ = (_e2430_ + (_e2430_ * (1f - (_e2427_ * _e2430_))));
                                        let _e2438_ = (_e2434_ + (_e2434_ * (1f - (_e2427_ * _e2434_))));
                                        let _e2442_ = (_e2438_ + (_e2438_ * (1f - (_e2427_ * _e2438_))));
                                        let _e2443_ = (_e2421_ + _e2395_);
                                        if (_e2443_ != _e2443_) {
                                            phi_12154_ = true;
                                        } else {
                                            phi_12154_ = (0.000001f >= _e2443_);
                                        }
                                        let _e2447_ = phi_12154_;
                                        let _e2448_ = select(_e2443_, 0.000001f, _e2447_);
                                        let _e2451_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2448_)));
                                        let _e2455_ = (_e2451_ + (_e2451_ * (1f - (_e2448_ * _e2451_))));
                                        let _e2459_ = (_e2455_ + (_e2455_ * (1f - (_e2448_ * _e2455_))));
                                        let _e2469_ = (_e2390_ <= 1000000000000000f);
                                        let _e2479_ = select(select(select(_e2390_, (0.5f * (_e2395_ + (_e2396_ * _e2442_))), _e2469_), select(-0.0005f, -(((0.0005f * (_e2395_ * (_e2459_ + (_e2459_ * (1f - (_e2448_ * _e2459_)))))) * (1f + (_e2395_ * _e2442_)))), _e2469_), (_e2373_ <= 0f)), 0f, (_e2390_ <= 0f));
                                        let _e2489_ = bitcast<u32>((((1f - (_e1554_ * _e2370_)) * _e2371_) + (_e2370_ * select(((_e2213_ * _e2479_) + (_e1554_ * (_e2383_ * (1f - (0.33333334f * (_e2383_ * _e2383_)))))), _e2479_, _e2326_))));
                                        let _e2490_ = (_e2489_ & 2139095040u);
                                        if (_e2490_ == 2139095040u) {
                                            phi_6059_ = 0u;
                                        } else {
                                            if (_e2490_ == 0u) {
                                                phi_6058_ = (_e2489_ & 2147483648u);
                                            } else {
                                                phi_6058_ = _e2489_;
                                            }
                                            let _e2495_ = phi_6058_;
                                            phi_6059_ = _e2495_;
                                        }
                                        let _e2497_ = phi_6059_;
                                        let _e2498_ = bitcast<f32>(_e2497_);
                                        let _e2500_ = (0.5f * (_e2342_ + _e2498_));
                                        let _e2507_ = (1024u + _e2166_);
                                        if (_e2507_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        scan_scratch[_e2507_] = (_e2342_ + (_e1573_ * (_e2500_ - _e2342_)));
                                        let _e2510_ = (1280u + _e2166_);
                                        if (_e2510_ < 1536u) {
                                        } else {
                                            phi_16756_ = true;
                                            break;
                                        }
                                        scan_scratch[_e2510_] = (_e2498_ + (_e1573_ * (_e2500_ - _e2498_)));
                                        phi_4435_ = _e2498_;
                                        phi_4437_ = _e2342_;
                                    } else {
                                        phi_4435_ = f32();
                                        phi_4437_ = f32();
                                    }
                                    let _e2514_ = phi_4435_;
                                    let _e2516_ = phi_4437_;
                                    phi_4431_ = _e2159_;
                                    phi_4434_ = _e2514_;
                                    phi_4436_ = _e2516_;
                                    phi_16756_ = _e1992_;
                                    if !(_e2165_) {
                                        break;
                                    }
                                }
                                let _e2519_ = phi_16756_;
                                phi_17036_ = _e2519_;
                                if _e2519_ {
                                    break;
                                }
                                if _e1659_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2521_ = local_13_;
                                local_10_[_e1347_] = _e2521_;
                                if _e1659_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2523_ = local_14_;
                                local_11_[_e1347_] = _e2523_;
                                let _e2524_ = local_10_;
                                let _e2530_ = local_11_;
                                phi_17048_ = _e2519_;
                                phi_6201_ = type_32_(_e1327_.member, _e1327_.member_1_, type_27_(type_24_(array<f32, 4>(_e2524_[0], _e2524_[1], _e2524_[2], _e2524_[3])), type_24_(array<f32, 4>(_e2530_[0], _e2530_[1], _e2530_[2], _e2530_[3]))), _e1327_.member_3_);
                            } else {
                                phi_17048_ = _e1992_;
                                phi_6201_ = _e1327_;
                            }
                            let _e2545_ = phi_17048_;
                            let _e2547_ = phi_6201_;
                            workgroupBarrier();
                            if _e931_ {
                                if _e2116_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2548_ = scan_scratch[_e2115_];
                                phi_6213_ = _e2548_;
                            } else {
                                phi_6213_ = 0f;
                            }
                            let _e2550_ = phi_6213_;
                            if _e931_ {
                                if _e2119_ {
                                } else {
                                    phi_17036_ = true;
                                    break;
                                }
                                let _e2551_ = scan_scratch[_e2118_];
                                phi_6225_ = _e2551_;
                            } else {
                                phi_6225_ = 0f;
                            }
                            let _e2553_ = phi_6225_;
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local[_e1347_] = _e2550_;
                            if _e1659_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            local_1_[_e1347_] = _e2553_;
                            let _e2557_ = (1f - (_e1594_ * _e2550_));
                            if (_e2557_ != _e2557_) {
                                phi_12334_ = true;
                            } else {
                                phi_12334_ = (0.05f >= _e2557_);
                            }
                            let _e2561_ = phi_12334_;
                            let _e2564_ = (1f - (_e1594_ * _e2553_));
                            if (_e2564_ != _e2564_) {
                                phi_12349_ = true;
                            } else {
                                phi_12349_ = (0.05f >= _e2564_);
                            }
                            let _e2568_ = phi_12349_;
                            if _e931_ {
                                let _e2574_ = bitcast<u32>((((_e2096_ * _e1601_) * select(_e2557_, 0.05f, _e2561_)) + (_e1608_ * _e2550_)));
                                let _e2575_ = (_e2574_ & 2139095040u);
                                if (_e2575_ == 2139095040u) {
                                    phi_6263_ = true;
                                } else {
                                    phi_6263_ = (_e2575_ == 0u);
                                }
                                let _e2579_ = phi_6263_;
                                phi_6269_ = bitcast<f32>(select(_e2574_, 0u, _e2579_));
                            } else {
                                phi_6269_ = 0f;
                            }
                            let _e2583_ = phi_6269_;
                            if _e931_ {
                                let _e2588_ = bitcast<u32>((((_e2095_ * _e1601_) * select(_e2564_, 0.05f, _e2568_)) + (_e1608_ * _e2553_)));
                                let _e2589_ = (_e2588_ & 2139095040u);
                                if (_e2589_ == 2139095040u) {
                                    phi_6287_ = true;
                                } else {
                                    phi_6287_ = (_e2589_ == 0u);
                                }
                                let _e2593_ = phi_6287_;
                                phi_6293_ = bitcast<f32>(select(_e2588_, 0u, _e2593_));
                            } else {
                                phi_6293_ = 0f;
                            }
                            let _e2597_ = phi_6293_;
                            if _e1287_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    lane_buf[_e114_].x = _e2583_;
                                    break;
                                }
                                case 1: {
                                    lane_buf[_e114_].y = _e2583_;
                                    break;
                                }
                                case 2: {
                                    lane_buf[_e114_].z = _e2583_;
                                    break;
                                }
                                default: {
                                    lane_buf[_e114_].w = _e2583_;
                                    break;
                                }
                            }
                            if _e1290_ {
                            } else {
                                phi_17036_ = true;
                                break;
                            }
                            switch bitcast<i32>(_e1347_) {
                                case 0: {
                                    lane_buf[_e1289_].x = _e2597_;
                                    break;
                                }
                                case 1: {
                                    lane_buf[_e1289_].y = _e2597_;
                                    break;
                                }
                                case 2: {
                                    lane_buf[_e1289_].z = _e2597_;
                                    break;
                                }
                                default: {
                                    lane_buf[_e1289_].w = _e2597_;
                                    break;
                                }
                            }
                            workgroupBarrier();
                            phi_17037_ = _e2545_;
                            phi_2418_ = _e2547_;
                        } else {
                            phi_17037_ = _e1323_;
                            phi_2418_ = type_32_();
                        }
                        let _e2617_ = phi_17037_;
                        let _e2619_ = phi_2418_;
                        phi_16474_ = _e2617_;
                        phi_2414_ = _e1340_;
                        phi_2417_ = _e2619_;
                        phi_17036_ = _e2617_;
                        if !(_e1346_) {
                            break;
                        }
                    }
                    let _e2622_ = phi_17036_;
                    phi_17053_ = _e2622_;
                    if _e2622_ {
                        break;
                    }
                    let _e2623_ = local_2_;
                    let _e2628_ = bitcast<u32>(_e2623_[0]);
                    let _e2629_ = (_e2628_ & 2139095040u);
                    if (_e2629_ == 2139095040u) {
                        phi_6347_ = 0u;
                    } else {
                        if (_e2629_ == 0u) {
                            phi_6346_ = (_e2628_ & 2147483648u);
                        } else {
                            phi_6346_ = _e2628_;
                        }
                        let _e2634_ = phi_6346_;
                        phi_6347_ = _e2634_;
                    }
                    let _e2636_ = phi_6347_;
                    let _e2638_ = bitcast<u32>(_e2623_[1]);
                    let _e2639_ = (_e2638_ & 2139095040u);
                    if (_e2639_ == 2139095040u) {
                        phi_6362_ = 0u;
                    } else {
                        if (_e2639_ == 0u) {
                            phi_6361_ = (_e2638_ & 2147483648u);
                        } else {
                            phi_6361_ = _e2638_;
                        }
                        let _e2644_ = phi_6361_;
                        phi_6362_ = _e2644_;
                    }
                    let _e2646_ = phi_6362_;
                    let _e2648_ = bitcast<u32>(_e2623_[2]);
                    let _e2649_ = (_e2648_ & 2139095040u);
                    if (_e2649_ == 2139095040u) {
                        phi_6377_ = 0u;
                    } else {
                        if (_e2649_ == 0u) {
                            phi_6376_ = (_e2648_ & 2147483648u);
                        } else {
                            phi_6376_ = _e2648_;
                        }
                        let _e2654_ = phi_6376_;
                        phi_6377_ = _e2654_;
                    }
                    let _e2656_ = phi_6377_;
                    let _e2658_ = bitcast<u32>(_e2623_[3]);
                    let _e2659_ = (_e2658_ & 2139095040u);
                    if (_e2659_ == 2139095040u) {
                        phi_6392_ = 0u;
                    } else {
                        if (_e2659_ == 0u) {
                            phi_6391_ = (_e2658_ & 2147483648u);
                        } else {
                            phi_6391_ = _e2658_;
                        }
                        let _e2664_ = phi_6391_;
                        phi_6392_ = _e2664_;
                    }
                    let _e2666_ = phi_6392_;
                    let _e2669_ = local_3_;
                    let _e2674_ = bitcast<u32>(_e2669_[0]);
                    let _e2675_ = (_e2674_ & 2139095040u);
                    if (_e2675_ == 2139095040u) {
                        phi_6415_ = 0u;
                    } else {
                        if (_e2675_ == 0u) {
                            phi_6414_ = (_e2674_ & 2147483648u);
                        } else {
                            phi_6414_ = _e2674_;
                        }
                        let _e2680_ = phi_6414_;
                        phi_6415_ = _e2680_;
                    }
                    let _e2682_ = phi_6415_;
                    let _e2684_ = bitcast<u32>(_e2669_[1]);
                    let _e2685_ = (_e2684_ & 2139095040u);
                    if (_e2685_ == 2139095040u) {
                        phi_6430_ = 0u;
                    } else {
                        if (_e2685_ == 0u) {
                            phi_6429_ = (_e2684_ & 2147483648u);
                        } else {
                            phi_6429_ = _e2684_;
                        }
                        let _e2690_ = phi_6429_;
                        phi_6430_ = _e2690_;
                    }
                    let _e2692_ = phi_6430_;
                    let _e2694_ = bitcast<u32>(_e2669_[2]);
                    let _e2695_ = (_e2694_ & 2139095040u);
                    if (_e2695_ == 2139095040u) {
                        phi_6445_ = 0u;
                    } else {
                        if (_e2695_ == 0u) {
                            phi_6444_ = (_e2694_ & 2147483648u);
                        } else {
                            phi_6444_ = _e2694_;
                        }
                        let _e2700_ = phi_6444_;
                        phi_6445_ = _e2700_;
                    }
                    let _e2702_ = phi_6445_;
                    let _e2704_ = bitcast<u32>(_e2669_[3]);
                    let _e2705_ = (_e2704_ & 2139095040u);
                    if (_e2705_ == 2139095040u) {
                        phi_6460_ = 0u;
                    } else {
                        if (_e2705_ == 0u) {
                            phi_6459_ = (_e2704_ & 2147483648u);
                        } else {
                            phi_6459_ = _e2704_;
                        }
                        let _e2710_ = phi_6459_;
                        phi_6460_ = _e2710_;
                    }
                    let _e2712_ = phi_6460_;
                    let _e2719_ = local_15_;
                    let _e2724_ = type_32_(type_27_(type_24_(array<f32, 4>(bitcast<f32>(_e2636_), bitcast<f32>(_e2646_), bitcast<f32>(_e2656_), bitcast<f32>(_e2666_))), type_24_(array<f32, 4>(bitcast<f32>(_e2682_), bitcast<f32>(_e2692_), bitcast<f32>(_e2702_), bitcast<f32>(_e2712_)))), _e2719_.member_1_, _e2719_.member_2_, _e2719_.member_3_);
                    let _e2725_ = local_4_;
                    let _e2730_ = bitcast<u32>(_e2725_[0]);
                    let _e2731_ = (_e2730_ & 2139095040u);
                    if (_e2731_ == 2139095040u) {
                        phi_6486_ = 0u;
                    } else {
                        if (_e2731_ == 0u) {
                            phi_6485_ = (_e2730_ & 2147483648u);
                        } else {
                            phi_6485_ = _e2730_;
                        }
                        let _e2736_ = phi_6485_;
                        phi_6486_ = _e2736_;
                    }
                    let _e2738_ = phi_6486_;
                    let _e2740_ = bitcast<u32>(_e2725_[1]);
                    let _e2741_ = (_e2740_ & 2139095040u);
                    if (_e2741_ == 2139095040u) {
                        phi_6501_ = 0u;
                    } else {
                        if (_e2741_ == 0u) {
                            phi_6500_ = (_e2740_ & 2147483648u);
                        } else {
                            phi_6500_ = _e2740_;
                        }
                        let _e2746_ = phi_6500_;
                        phi_6501_ = _e2746_;
                    }
                    let _e2748_ = phi_6501_;
                    let _e2750_ = bitcast<u32>(_e2725_[2]);
                    let _e2751_ = (_e2750_ & 2139095040u);
                    if (_e2751_ == 2139095040u) {
                        phi_6516_ = 0u;
                    } else {
                        if (_e2751_ == 0u) {
                            phi_6515_ = (_e2750_ & 2147483648u);
                        } else {
                            phi_6515_ = _e2750_;
                        }
                        let _e2756_ = phi_6515_;
                        phi_6516_ = _e2756_;
                    }
                    let _e2758_ = phi_6516_;
                    let _e2760_ = bitcast<u32>(_e2725_[3]);
                    let _e2761_ = (_e2760_ & 2139095040u);
                    if (_e2761_ == 2139095040u) {
                        phi_6531_ = 0u;
                    } else {
                        if (_e2761_ == 0u) {
                            phi_6530_ = (_e2760_ & 2147483648u);
                        } else {
                            phi_6530_ = _e2760_;
                        }
                        let _e2766_ = phi_6530_;
                        phi_6531_ = _e2766_;
                    }
                    let _e2768_ = phi_6531_;
                    let _e2771_ = local_5_;
                    let _e2776_ = bitcast<u32>(_e2771_[0]);
                    let _e2777_ = (_e2776_ & 2139095040u);
                    if (_e2777_ == 2139095040u) {
                        phi_6554_ = 0u;
                    } else {
                        if (_e2777_ == 0u) {
                            phi_6553_ = (_e2776_ & 2147483648u);
                        } else {
                            phi_6553_ = _e2776_;
                        }
                        let _e2782_ = phi_6553_;
                        phi_6554_ = _e2782_;
                    }
                    let _e2784_ = phi_6554_;
                    let _e2786_ = bitcast<u32>(_e2771_[1]);
                    let _e2787_ = (_e2786_ & 2139095040u);
                    if (_e2787_ == 2139095040u) {
                        phi_6569_ = 0u;
                    } else {
                        if (_e2787_ == 0u) {
                            phi_6568_ = (_e2786_ & 2147483648u);
                        } else {
                            phi_6568_ = _e2786_;
                        }
                        let _e2792_ = phi_6568_;
                        phi_6569_ = _e2792_;
                    }
                    let _e2794_ = phi_6569_;
                    let _e2796_ = bitcast<u32>(_e2771_[2]);
                    let _e2797_ = (_e2796_ & 2139095040u);
                    if (_e2797_ == 2139095040u) {
                        phi_6584_ = 0u;
                    } else {
                        if (_e2797_ == 0u) {
                            phi_6583_ = (_e2796_ & 2147483648u);
                        } else {
                            phi_6583_ = _e2796_;
                        }
                        let _e2802_ = phi_6583_;
                        phi_6584_ = _e2802_;
                    }
                    let _e2804_ = phi_6584_;
                    let _e2806_ = bitcast<u32>(_e2771_[3]);
                    let _e2807_ = (_e2806_ & 2139095040u);
                    if (_e2807_ == 2139095040u) {
                        phi_6599_ = 0u;
                    } else {
                        if (_e2807_ == 0u) {
                            phi_6598_ = (_e2806_ & 2147483648u);
                        } else {
                            phi_6598_ = _e2806_;
                        }
                        let _e2812_ = phi_6598_;
                        phi_6599_ = _e2812_;
                    }
                    let _e2814_ = phi_6599_;
                    let _e2824_ = type_32_(_e2724_.member, type_27_(type_24_(array<f32, 4>(bitcast<f32>(_e2738_), bitcast<f32>(_e2748_), bitcast<f32>(_e2758_), bitcast<f32>(_e2768_))), type_24_(array<f32, 4>(bitcast<f32>(_e2784_), bitcast<f32>(_e2794_), bitcast<f32>(_e2804_), bitcast<f32>(_e2814_)))), _e2724_.member_2_, _e2724_.member_3_);
                    if _e1287_ {
                    } else {
                        phi_17053_ = true;
                        break;
                    }
                    let _e2825_ = lane_buf[_e114_];
                    if _e1290_ {
                    } else {
                        phi_17053_ = true;
                        break;
                    }
                    let _e2830_ = lane_buf[_e1289_];
                    let _e2836_ = local_16_;
                    if _e932_ {
                        phi_6670_ = _e2836_.member_3_.member_1_;
                        phi_6671_ = _e2836_.member_3_.member;
                    } else {
                        let _e2841_ = ((_e114_ - 1u) & 255u);
                        if (_e2841_ < 512u) {
                        } else {
                            phi_17053_ = true;
                            break;
                        }
                        let _e2844_ = lane_buf[_e2841_];
                        let _e2851_ = (256u + _e2841_);
                        if (_e2851_ < 512u) {
                        } else {
                            phi_17053_ = true;
                            break;
                        }
                        let _e2854_ = lane_buf[_e2851_];
                        phi_6670_ = type_24_(array<f32, 4>(_e2854_.x, _e2854_.y, _e2854_.z, _e2854_.w));
                        phi_6671_ = type_24_(array<f32, 4>(_e2844_.x, _e2844_.y, _e2844_.z, _e2844_.w));
                    }
                    let _e2862_ = phi_6670_;
                    let _e2864_ = phi_6671_;
                    let _e2886_ = select(0f, _e2864_.member[0], (bitcast<f32>((bitcast<u32>(_e2864_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2888_ = select(0f, _e2864_.member[1], (bitcast<f32>((bitcast<u32>(_e2864_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2890_ = select(0f, _e2864_.member[2], (bitcast<f32>((bitcast<u32>(_e2864_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2892_ = select(0f, _e2864_.member[3], (bitcast<f32>((bitcast<u32>(_e2864_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2893_ = (_e2886_ != _e2886_);
                    if _e2893_ {
                        phi_12364_ = true;
                    } else {
                        phi_12364_ = (-1000000000000000000f >= _e2886_);
                    }
                    let _e2896_ = phi_12364_;
                    let _e2897_ = select(_e2886_, -1000000000000000000f, _e2896_);
                    let _e2898_ = (_e2888_ != _e2888_);
                    if _e2898_ {
                        phi_12379_ = true;
                    } else {
                        phi_12379_ = (-1000000000000000000f >= _e2888_);
                    }
                    let _e2901_ = phi_12379_;
                    let _e2902_ = select(_e2888_, -1000000000000000000f, _e2901_);
                    let _e2903_ = (_e2890_ != _e2890_);
                    if _e2903_ {
                        phi_12394_ = true;
                    } else {
                        phi_12394_ = (-1000000000000000000f >= _e2890_);
                    }
                    let _e2906_ = phi_12394_;
                    let _e2907_ = select(_e2890_, -1000000000000000000f, _e2906_);
                    let _e2908_ = (_e2892_ != _e2892_);
                    if _e2908_ {
                        phi_12409_ = true;
                    } else {
                        phi_12409_ = (-1000000000000000000f >= _e2892_);
                    }
                    let _e2911_ = phi_12409_;
                    let _e2912_ = select(_e2892_, -1000000000000000000f, _e2911_);
                    if (_e2897_ != _e2897_) {
                        phi_12424_ = true;
                    } else {
                        phi_12424_ = (1000000000000000000f <= _e2897_);
                    }
                    let _e2916_ = phi_12424_;
                    let _e2917_ = select(_e2897_, 1000000000000000000f, _e2916_);
                    if (_e2902_ != _e2902_) {
                        phi_12439_ = true;
                    } else {
                        phi_12439_ = (1000000000000000000f <= _e2902_);
                    }
                    let _e2921_ = phi_12439_;
                    let _e2922_ = select(_e2902_, 1000000000000000000f, _e2921_);
                    if (_e2907_ != _e2907_) {
                        phi_12454_ = true;
                    } else {
                        phi_12454_ = (1000000000000000000f <= _e2907_);
                    }
                    let _e2926_ = phi_12454_;
                    let _e2927_ = select(_e2907_, 1000000000000000000f, _e2926_);
                    if (_e2912_ != _e2912_) {
                        phi_12469_ = true;
                    } else {
                        phi_12469_ = (1000000000000000000f <= _e2912_);
                    }
                    let _e2931_ = phi_12469_;
                    let _e2932_ = select(_e2912_, 1000000000000000000f, _e2931_);
                    let _e2946_ = select(0f, _e2825_.x, (bitcast<f32>((bitcast<u32>(_e2825_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2948_ = select(0f, _e2825_.y, (bitcast<f32>((bitcast<u32>(_e2825_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2950_ = select(0f, _e2825_.z, (bitcast<f32>((bitcast<u32>(_e2825_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2952_ = select(0f, _e2825_.w, (bitcast<f32>((bitcast<u32>(_e2825_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2953_ = (_e2946_ != _e2946_);
                    if _e2953_ {
                        phi_12484_ = true;
                    } else {
                        phi_12484_ = (-1000000000000000000f >= _e2946_);
                    }
                    let _e2956_ = phi_12484_;
                    let _e2957_ = select(_e2946_, -1000000000000000000f, _e2956_);
                    let _e2958_ = (_e2948_ != _e2948_);
                    if _e2958_ {
                        phi_12499_ = true;
                    } else {
                        phi_12499_ = (-1000000000000000000f >= _e2948_);
                    }
                    let _e2961_ = phi_12499_;
                    let _e2962_ = select(_e2948_, -1000000000000000000f, _e2961_);
                    let _e2963_ = (_e2950_ != _e2950_);
                    if _e2963_ {
                        phi_12514_ = true;
                    } else {
                        phi_12514_ = (-1000000000000000000f >= _e2950_);
                    }
                    let _e2966_ = phi_12514_;
                    let _e2967_ = select(_e2950_, -1000000000000000000f, _e2966_);
                    let _e2968_ = (_e2952_ != _e2952_);
                    if _e2968_ {
                        phi_12529_ = true;
                    } else {
                        phi_12529_ = (-1000000000000000000f >= _e2952_);
                    }
                    let _e2971_ = phi_12529_;
                    let _e2972_ = select(_e2952_, -1000000000000000000f, _e2971_);
                    if (_e2957_ != _e2957_) {
                        phi_12544_ = true;
                    } else {
                        phi_12544_ = (1000000000000000000f <= _e2957_);
                    }
                    let _e2976_ = phi_12544_;
                    let _e2977_ = select(_e2957_, 1000000000000000000f, _e2976_);
                    if (_e2962_ != _e2962_) {
                        phi_12559_ = true;
                    } else {
                        phi_12559_ = (1000000000000000000f <= _e2962_);
                    }
                    let _e2981_ = phi_12559_;
                    let _e2982_ = select(_e2962_, 1000000000000000000f, _e2981_);
                    if (_e2967_ != _e2967_) {
                        phi_12574_ = true;
                    } else {
                        phi_12574_ = (1000000000000000000f <= _e2967_);
                    }
                    let _e2986_ = phi_12574_;
                    let _e2987_ = select(_e2967_, 1000000000000000000f, _e2986_);
                    if (_e2972_ != _e2972_) {
                        phi_12589_ = true;
                    } else {
                        phi_12589_ = (1000000000000000000f <= _e2972_);
                    }
                    let _e2991_ = phi_12589_;
                    let _e2992_ = select(_e2972_, 1000000000000000000f, _e2991_);
                    let _e2993_ = array<f32, 4>(_e2977_, _e2982_, _e2987_, _e2992_);
                    let _e2994_ = (_e2917_ != _e2917_);
                    if _e2994_ {
                        phi_12604_ = true;
                    } else {
                        phi_12604_ = (_e2977_ <= _e2917_);
                    }
                    let _e2997_ = phi_12604_;
                    let _e2998_ = select(_e2917_, _e2977_, _e2997_);
                    let _e2999_ = (_e2922_ != _e2922_);
                    if _e2999_ {
                        phi_12619_ = true;
                    } else {
                        phi_12619_ = (_e2982_ <= _e2922_);
                    }
                    let _e3002_ = phi_12619_;
                    let _e3003_ = select(_e2922_, _e2982_, _e3002_);
                    let _e3004_ = (_e2927_ != _e2927_);
                    if _e3004_ {
                        phi_12634_ = true;
                    } else {
                        phi_12634_ = (_e2987_ <= _e2927_);
                    }
                    let _e3007_ = phi_12634_;
                    let _e3008_ = select(_e2927_, _e2987_, _e3007_);
                    let _e3009_ = (_e2932_ != _e2932_);
                    if _e3009_ {
                        phi_12649_ = true;
                    } else {
                        phi_12649_ = (_e2992_ <= _e2932_);
                    }
                    let _e3012_ = phi_12649_;
                    let _e3013_ = select(_e2932_, _e2992_, _e3012_);
                    if _e2994_ {
                        phi_12664_ = true;
                    } else {
                        phi_12664_ = (_e2977_ >= _e2917_);
                    }
                    let _e3016_ = phi_12664_;
                    let _e3017_ = select(_e2917_, _e2977_, _e3016_);
                    if _e2999_ {
                        phi_12679_ = true;
                    } else {
                        phi_12679_ = (_e2982_ >= _e2922_);
                    }
                    let _e3020_ = phi_12679_;
                    let _e3021_ = select(_e2922_, _e2982_, _e3020_);
                    if _e3004_ {
                        phi_12694_ = true;
                    } else {
                        phi_12694_ = (_e2987_ >= _e2927_);
                    }
                    let _e3024_ = phi_12694_;
                    let _e3025_ = select(_e2927_, _e2987_, _e3024_);
                    if _e3009_ {
                        phi_12709_ = true;
                    } else {
                        phi_12709_ = (_e2992_ >= _e2932_);
                    }
                    let _e3028_ = phi_12709_;
                    let _e3029_ = select(_e2932_, _e2992_, _e3028_);
                    let _e3030_ = (_e2998_ != _e2998_);
                    if _e3030_ {
                        phi_12724_ = true;
                    } else {
                        phi_12724_ = (-1f >= _e2998_);
                    }
                    let _e3033_ = phi_12724_;
                    let _e3034_ = select(_e2998_, -1f, _e3033_);
                    let _e3035_ = (_e3003_ != _e3003_);
                    if _e3035_ {
                        phi_12739_ = true;
                    } else {
                        phi_12739_ = (-1f >= _e3003_);
                    }
                    let _e3038_ = phi_12739_;
                    let _e3039_ = select(_e3003_, -1f, _e3038_);
                    let _e3040_ = (_e3008_ != _e3008_);
                    if _e3040_ {
                        phi_12754_ = true;
                    } else {
                        phi_12754_ = (-1f >= _e3008_);
                    }
                    let _e3043_ = phi_12754_;
                    let _e3044_ = select(_e3008_, -1f, _e3043_);
                    let _e3045_ = (_e3013_ != _e3013_);
                    if _e3045_ {
                        phi_12769_ = true;
                    } else {
                        phi_12769_ = (-1f >= _e3013_);
                    }
                    let _e3048_ = phi_12769_;
                    let _e3049_ = select(_e3013_, -1f, _e3048_);
                    if (_e3034_ != _e3034_) {
                        phi_12784_ = true;
                    } else {
                        phi_12784_ = (1f <= _e3034_);
                    }
                    let _e3053_ = phi_12784_;
                    let _e3054_ = select(_e3034_, 1f, _e3053_);
                    if (_e3039_ != _e3039_) {
                        phi_12799_ = true;
                    } else {
                        phi_12799_ = (1f <= _e3039_);
                    }
                    let _e3058_ = phi_12799_;
                    let _e3059_ = select(_e3039_, 1f, _e3058_);
                    if (_e3044_ != _e3044_) {
                        phi_12814_ = true;
                    } else {
                        phi_12814_ = (1f <= _e3044_);
                    }
                    let _e3063_ = phi_12814_;
                    let _e3064_ = select(_e3044_, 1f, _e3063_);
                    if (_e3049_ != _e3049_) {
                        phi_12829_ = true;
                    } else {
                        phi_12829_ = (1f <= _e3049_);
                    }
                    let _e3068_ = phi_12829_;
                    let _e3069_ = select(_e3049_, 1f, _e3068_);
                    let _e3070_ = (_e3017_ != _e3017_);
                    if _e3070_ {
                        phi_12844_ = true;
                    } else {
                        phi_12844_ = (-1f >= _e3017_);
                    }
                    let _e3073_ = phi_12844_;
                    let _e3074_ = select(_e3017_, -1f, _e3073_);
                    let _e3075_ = (_e3021_ != _e3021_);
                    if _e3075_ {
                        phi_12859_ = true;
                    } else {
                        phi_12859_ = (-1f >= _e3021_);
                    }
                    let _e3078_ = phi_12859_;
                    let _e3079_ = select(_e3021_, -1f, _e3078_);
                    let _e3080_ = (_e3025_ != _e3025_);
                    if _e3080_ {
                        phi_12874_ = true;
                    } else {
                        phi_12874_ = (-1f >= _e3025_);
                    }
                    let _e3083_ = phi_12874_;
                    let _e3084_ = select(_e3025_, -1f, _e3083_);
                    let _e3085_ = (_e3029_ != _e3029_);
                    if _e3085_ {
                        phi_12889_ = true;
                    } else {
                        phi_12889_ = (-1f >= _e3029_);
                    }
                    let _e3088_ = phi_12889_;
                    let _e3089_ = select(_e3029_, -1f, _e3088_);
                    if (_e3074_ != _e3074_) {
                        phi_12904_ = true;
                    } else {
                        phi_12904_ = (1f <= _e3074_);
                    }
                    let _e3093_ = phi_12904_;
                    let _e3094_ = select(_e3074_, 1f, _e3093_);
                    if (_e3079_ != _e3079_) {
                        phi_12919_ = true;
                    } else {
                        phi_12919_ = (1f <= _e3079_);
                    }
                    let _e3098_ = phi_12919_;
                    let _e3099_ = select(_e3079_, 1f, _e3098_);
                    if (_e3084_ != _e3084_) {
                        phi_12934_ = true;
                    } else {
                        phi_12934_ = (1f <= _e3084_);
                    }
                    let _e3103_ = phi_12934_;
                    let _e3104_ = select(_e3084_, 1f, _e3103_);
                    if (_e3089_ != _e3089_) {
                        phi_12949_ = true;
                    } else {
                        phi_12949_ = (1f <= _e3089_);
                    }
                    let _e3108_ = phi_12949_;
                    let _e3109_ = select(_e3089_, 1f, _e3108_);
                    let _e3114_ = (0.5f * (_e3094_ + _e3054_));
                    let _e3115_ = (0.5f * (_e3099_ + _e3059_));
                    let _e3116_ = (0.5f * (_e3104_ + _e3064_));
                    let _e3117_ = (0.5f * (_e3109_ + _e3069_));
                    let _e3118_ = (_e3094_ - _e3054_);
                    let _e3119_ = (_e3099_ - _e3059_);
                    let _e3120_ = (_e3104_ - _e3064_);
                    let _e3121_ = (_e3109_ - _e3069_);
                    let _e3122_ = (0.5f * _e3118_);
                    let _e3123_ = (0.5f * _e3119_);
                    let _e3124_ = (0.5f * _e3120_);
                    let _e3125_ = (0.5f * _e3121_);
                    let _e3126_ = (_e3114_ * _e3114_);
                    let _e3127_ = (_e3115_ * _e3115_);
                    let _e3128_ = (_e3116_ * _e3116_);
                    let _e3129_ = (_e3117_ * _e3117_);
                    let _e3130_ = (_e3122_ * _e3122_);
                    let _e3131_ = (_e3123_ * _e3123_);
                    let _e3132_ = (_e3124_ * _e3124_);
                    let _e3133_ = (_e3125_ * _e3125_);
                    if (_e3118_ != _e3118_) {
                        phi_12964_ = true;
                    } else {
                        phi_12964_ = (0f >= _e3118_);
                    }
                    let _e3137_ = phi_12964_;
                    let _e3138_ = select(_e3118_, 0f, _e3137_);
                    if (_e3119_ != _e3119_) {
                        phi_12979_ = true;
                    } else {
                        phi_12979_ = (0f >= _e3119_);
                    }
                    let _e3142_ = phi_12979_;
                    let _e3143_ = select(_e3119_, 0f, _e3142_);
                    if (_e3120_ != _e3120_) {
                        phi_12994_ = true;
                    } else {
                        phi_12994_ = (0f >= _e3120_);
                    }
                    let _e3147_ = phi_12994_;
                    let _e3148_ = select(_e3120_, 0f, _e3147_);
                    if (_e3121_ != _e3121_) {
                        phi_13009_ = true;
                    } else {
                        phi_13009_ = (0f >= _e3121_);
                    }
                    let _e3152_ = phi_13009_;
                    let _e3153_ = select(_e3121_, 0f, _e3152_);
                    let _e3154_ = (_e3017_ - _e2998_);
                    let _e3155_ = (_e3021_ - _e3003_);
                    let _e3156_ = (_e3025_ - _e3008_);
                    let _e3157_ = (_e3029_ - _e3013_);
                    if _e3070_ {
                        phi_13024_ = true;
                    } else {
                        phi_13024_ = (-1f <= _e3017_);
                    }
                    let _e3160_ = phi_13024_;
                    if _e3075_ {
                        phi_13039_ = true;
                    } else {
                        phi_13039_ = (-1f <= _e3021_);
                    }
                    let _e3164_ = phi_13039_;
                    if _e3080_ {
                        phi_13054_ = true;
                    } else {
                        phi_13054_ = (-1f <= _e3025_);
                    }
                    let _e3168_ = phi_13054_;
                    if _e3085_ {
                        phi_13069_ = true;
                    } else {
                        phi_13069_ = (-1f <= _e3029_);
                    }
                    let _e3172_ = phi_13069_;
                    let _e3174_ = (select(_e3017_, -1f, _e3160_) - _e2998_);
                    let _e3175_ = (select(_e3021_, -1f, _e3164_) - _e3003_);
                    let _e3176_ = (select(_e3025_, -1f, _e3168_) - _e3008_);
                    let _e3177_ = (select(_e3029_, -1f, _e3172_) - _e3013_);
                    if (_e3174_ != _e3174_) {
                        phi_13084_ = true;
                    } else {
                        phi_13084_ = (0f >= _e3174_);
                    }
                    let _e3181_ = phi_13084_;
                    if (_e3175_ != _e3175_) {
                        phi_13099_ = true;
                    } else {
                        phi_13099_ = (0f >= _e3175_);
                    }
                    let _e3186_ = phi_13099_;
                    if (_e3176_ != _e3176_) {
                        phi_13114_ = true;
                    } else {
                        phi_13114_ = (0f >= _e3176_);
                    }
                    let _e3191_ = phi_13114_;
                    if (_e3177_ != _e3177_) {
                        phi_13129_ = true;
                    } else {
                        phi_13129_ = (0f >= _e3177_);
                    }
                    let _e3196_ = phi_13129_;
                    if _e3030_ {
                        phi_13144_ = true;
                    } else {
                        phi_13144_ = (1f >= _e2998_);
                    }
                    let _e3200_ = phi_13144_;
                    if _e3035_ {
                        phi_13159_ = true;
                    } else {
                        phi_13159_ = (1f >= _e3003_);
                    }
                    let _e3204_ = phi_13159_;
                    if _e3040_ {
                        phi_13174_ = true;
                    } else {
                        phi_13174_ = (1f >= _e3008_);
                    }
                    let _e3208_ = phi_13174_;
                    if _e3045_ {
                        phi_13189_ = true;
                    } else {
                        phi_13189_ = (1f >= _e3013_);
                    }
                    let _e3212_ = phi_13189_;
                    let _e3214_ = (_e3017_ - select(_e2998_, 1f, _e3200_));
                    let _e3215_ = (_e3021_ - select(_e3003_, 1f, _e3204_));
                    let _e3216_ = (_e3025_ - select(_e3008_, 1f, _e3208_));
                    let _e3217_ = (_e3029_ - select(_e3013_, 1f, _e3212_));
                    if (_e3214_ != _e3214_) {
                        phi_13204_ = true;
                    } else {
                        phi_13204_ = (0f >= _e3214_);
                    }
                    let _e3221_ = phi_13204_;
                    if (_e3215_ != _e3215_) {
                        phi_13219_ = true;
                    } else {
                        phi_13219_ = (0f >= _e3215_);
                    }
                    let _e3226_ = phi_13219_;
                    if (_e3216_ != _e3216_) {
                        phi_13234_ = true;
                    } else {
                        phi_13234_ = (0f >= _e3216_);
                    }
                    let _e3231_ = phi_13234_;
                    if (_e3217_ != _e3217_) {
                        phi_13249_ = true;
                    } else {
                        phi_13249_ = (0f >= _e3217_);
                    }
                    let _e3236_ = phi_13249_;
                    let _e3240_ = bitcast<f32>((bitcast<u32>(_e2917_) & 2147483647u));
                    let _e3243_ = bitcast<f32>((bitcast<u32>(_e2922_) & 2147483647u));
                    let _e3246_ = bitcast<f32>((bitcast<u32>(_e2927_) & 2147483647u));
                    let _e3249_ = bitcast<f32>((bitcast<u32>(_e2932_) & 2147483647u));
                    let _e3252_ = bitcast<f32>((bitcast<u32>(_e2977_) & 2147483647u));
                    let _e3255_ = bitcast<f32>((bitcast<u32>(_e2982_) & 2147483647u));
                    let _e3258_ = bitcast<f32>((bitcast<u32>(_e2987_) & 2147483647u));
                    let _e3261_ = bitcast<f32>((bitcast<u32>(_e2992_) & 2147483647u));
                    if (_e3240_ != _e3240_) {
                        phi_13264_ = true;
                    } else {
                        phi_13264_ = (_e3252_ >= _e3240_);
                    }
                    let _e3265_ = phi_13264_;
                    let _e3266_ = select(_e3240_, _e3252_, _e3265_);
                    if (_e3243_ != _e3243_) {
                        phi_13279_ = true;
                    } else {
                        phi_13279_ = (_e3255_ >= _e3243_);
                    }
                    let _e3270_ = phi_13279_;
                    let _e3271_ = select(_e3243_, _e3255_, _e3270_);
                    if (_e3246_ != _e3246_) {
                        phi_13294_ = true;
                    } else {
                        phi_13294_ = (_e3258_ >= _e3246_);
                    }
                    let _e3275_ = phi_13294_;
                    let _e3276_ = select(_e3246_, _e3258_, _e3275_);
                    if (_e3249_ != _e3249_) {
                        phi_13309_ = true;
                    } else {
                        phi_13309_ = (_e3261_ >= _e3249_);
                    }
                    let _e3280_ = phi_13309_;
                    let _e3281_ = select(_e3249_, _e3261_, _e3280_);
                    let _e3298_ = (_e3266_ <= 0f);
                    let _e3299_ = select((_e3114_ * (1.5f - (0.5f * (_e3126_ + _e3130_)))), 0f, _e3298_);
                    let _e3300_ = (_e3271_ <= 0f);
                    let _e3301_ = select((_e3115_ * (1.5f - (0.5f * (_e3127_ + _e3131_)))), 0f, _e3300_);
                    let _e3302_ = (_e3276_ <= 0f);
                    let _e3303_ = select((_e3116_ * (1.5f - (0.5f * (_e3128_ + _e3132_)))), 0f, _e3302_);
                    let _e3304_ = (_e3281_ <= 0f);
                    let _e3305_ = select((_e3117_ * (1.5f - (0.5f * (_e3129_ + _e3133_)))), 0f, _e3304_);
                    let _e3347_ = select(((_e3126_ + (0.33333334f * _e3130_)) - (((_e3126_ * _e3126_) + ((2f * _e3126_) * _e3130_)) + (0.2f * (_e3130_ * _e3130_)))), 0f, _e3298_);
                    let _e3348_ = select(((_e3127_ + (0.33333334f * _e3131_)) - (((_e3127_ * _e3127_) + ((2f * _e3127_) * _e3131_)) + (0.2f * (_e3131_ * _e3131_)))), 0f, _e3300_);
                    let _e3349_ = select(((_e3128_ + (0.33333334f * _e3132_)) - (((_e3128_ * _e3128_) + ((2f * _e3128_) * _e3132_)) + (0.2f * (_e3132_ * _e3132_)))), 0f, _e3302_);
                    let _e3350_ = select(((_e3129_ + (0.33333334f * _e3133_)) - (((_e3129_ * _e3129_) + ((2f * _e3129_) * _e3133_)) + (0.2f * (_e3133_ * _e3133_)))), 0f, _e3304_);
                    if (_e3266_ <= 1f) {
                        if (_e3271_ <= 1f) {
                            let _e3354_ = (_e3276_ <= 1f);
                            if _e3354_ {
                                phi_7106_ = (_e3281_ <= 1f);
                            } else {
                                phi_7106_ = bool();
                            }
                            let _e3357_ = phi_7106_;
                            phi_7108_ = _e3357_;
                            phi_7109_ = select(true, false, _e3354_);
                        } else {
                            phi_7108_ = bool();
                            phi_7109_ = true;
                        }
                        let _e3360_ = phi_7108_;
                        let _e3362_ = phi_7109_;
                        phi_7110_ = _e3360_;
                        phi_7111_ = _e3362_;
                    } else {
                        phi_7110_ = bool();
                        phi_7111_ = true;
                    }
                    let _e3364_ = phi_7110_;
                    let _e3366_ = phi_7111_;
                    if select(_e3364_, false, _e3366_) {
                        phi_7489_ = type_33_(type_24_(array<f32, 4>(_e3299_, _e3301_, _e3303_, _e3305_)), type_24_(array<f32, 4>(_e3347_, _e3348_, _e3349_, _e3350_)), type_24_(_e2993_));
                    } else {
                        if (_e3154_ != _e3154_) {
                            phi_13324_ = true;
                        } else {
                            phi_13324_ = (0.000001f >= _e3154_);
                        }
                        let _e3371_ = phi_13324_;
                        let _e3372_ = select(_e3154_, 0.000001f, _e3371_);
                        if (_e3155_ != _e3155_) {
                            phi_13339_ = true;
                        } else {
                            phi_13339_ = (0.000001f >= _e3155_);
                        }
                        let _e3376_ = phi_13339_;
                        let _e3377_ = select(_e3155_, 0.000001f, _e3376_);
                        if (_e3156_ != _e3156_) {
                            phi_13354_ = true;
                        } else {
                            phi_13354_ = (0.000001f >= _e3156_);
                        }
                        let _e3381_ = phi_13354_;
                        let _e3382_ = select(_e3156_, 0.000001f, _e3381_);
                        if (_e3157_ != _e3157_) {
                            phi_13369_ = true;
                        } else {
                            phi_13369_ = (0.000001f >= _e3157_);
                        }
                        let _e3386_ = phi_13369_;
                        let _e3387_ = select(_e3157_, 0.000001f, _e3386_);
                        let _e3390_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3372_)));
                        let _e3393_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3377_)));
                        let _e3396_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3382_)));
                        let _e3399_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3387_)));
                        let _e3409_ = (_e3390_ + (_e3390_ * (1f - (_e3372_ * _e3390_))));
                        let _e3411_ = (_e3393_ + (_e3393_ * (1f - (_e3377_ * _e3393_))));
                        let _e3413_ = (_e3396_ + (_e3396_ * (1f - (_e3382_ * _e3396_))));
                        let _e3415_ = (_e3399_ + (_e3399_ * (1f - (_e3387_ * _e3399_))));
                        let _e3425_ = (_e3409_ + (_e3409_ * (1f - (_e3372_ * _e3409_))));
                        let _e3427_ = (_e3411_ + (_e3411_ * (1f - (_e3377_ * _e3411_))));
                        let _e3429_ = (_e3413_ + (_e3413_ * (1f - (_e3382_ * _e3413_))));
                        let _e3431_ = (_e3415_ + (_e3415_ * (1f - (_e3387_ * _e3415_))));
                        let _e3484_ = (select((select(0f, 1f, (-1f <= _e2998_)) * select(0f, 1f, (_e3017_ <= 1f))), 1f, (_e3154_ <= 0.000001f)) <= 0f);
                        let _e3486_ = (select((select(0f, 1f, (-1f <= _e3003_)) * select(0f, 1f, (_e3021_ <= 1f))), 1f, (_e3155_ <= 0.000001f)) <= 0f);
                        let _e3488_ = (select((select(0f, 1f, (-1f <= _e3008_)) * select(0f, 1f, (_e3025_ <= 1f))), 1f, (_e3156_ <= 0.000001f)) <= 0f);
                        let _e3490_ = (select((select(0f, 1f, (-1f <= _e3013_)) * select(0f, 1f, (_e3029_ <= 1f))), 1f, (_e3157_ <= 0.000001f)) <= 0f);
                        let _e3496_ = (1f <= _e2998_);
                        let _e3498_ = (1f <= _e3003_);
                        let _e3500_ = (1f <= _e3008_);
                        let _e3502_ = (1f <= _e3013_);
                        let _e3504_ = (_e3017_ <= -1f);
                        let _e3506_ = (_e3021_ <= -1f);
                        let _e3508_ = (_e3025_ <= -1f);
                        let _e3510_ = (_e3029_ <= -1f);
                        phi_7489_ = type_33_(type_24_(array<f32, 4>(select(select(select(_e3299_, (((select(_e3214_, 0f, _e3221_) - select(_e3174_, 0f, _e3181_)) + (_e3138_ * _e3299_)) * _e3425_), _e3484_), 1f, _e3496_), -1f, _e3504_), select(select(select(_e3301_, (((select(_e3215_, 0f, _e3226_) - select(_e3175_, 0f, _e3186_)) + (_e3143_ * _e3301_)) * _e3427_), _e3486_), 1f, _e3498_), -1f, _e3506_), select(select(select(_e3303_, (((select(_e3216_, 0f, _e3231_) - select(_e3176_, 0f, _e3191_)) + (_e3148_ * _e3303_)) * _e3429_), _e3488_), 1f, _e3500_), -1f, _e3508_), select(select(select(_e3305_, (((select(_e3217_, 0f, _e3236_) - select(_e3177_, 0f, _e3196_)) + (_e3153_ * _e3305_)) * _e3431_), _e3490_), 1f, _e3502_), -1f, _e3510_))), type_24_(array<f32, 4>(select(select(select(_e3347_, ((_e3138_ * _e3347_) * _e3425_), _e3484_), 0f, _e3496_), 0f, _e3504_), select(select(select(_e3348_, ((_e3143_ * _e3348_) * _e3427_), _e3486_), 0f, _e3498_), 0f, _e3506_), select(select(select(_e3349_, ((_e3148_ * _e3349_) * _e3429_), _e3488_), 0f, _e3500_), 0f, _e3508_), select(select(select(_e3350_, ((_e3153_ * _e3350_) * _e3431_), _e3490_), 0f, _e3502_), 0f, _e3510_))), type_24_(_e2993_));
                    }
                    let _e3531_ = phi_7489_;
                    let _e3554_ = select(0f, _e2862_.member[0], (bitcast<f32>((bitcast<u32>(_e2862_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3556_ = select(0f, _e2862_.member[1], (bitcast<f32>((bitcast<u32>(_e2862_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3558_ = select(0f, _e2862_.member[2], (bitcast<f32>((bitcast<u32>(_e2862_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3560_ = select(0f, _e2862_.member[3], (bitcast<f32>((bitcast<u32>(_e2862_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3561_ = (_e3554_ != _e3554_);
                    if _e3561_ {
                        phi_13384_ = true;
                    } else {
                        phi_13384_ = (-1000000000000000000f >= _e3554_);
                    }
                    let _e3564_ = phi_13384_;
                    let _e3565_ = select(_e3554_, -1000000000000000000f, _e3564_);
                    let _e3566_ = (_e3556_ != _e3556_);
                    if _e3566_ {
                        phi_13399_ = true;
                    } else {
                        phi_13399_ = (-1000000000000000000f >= _e3556_);
                    }
                    let _e3569_ = phi_13399_;
                    let _e3570_ = select(_e3556_, -1000000000000000000f, _e3569_);
                    let _e3571_ = (_e3558_ != _e3558_);
                    if _e3571_ {
                        phi_13414_ = true;
                    } else {
                        phi_13414_ = (-1000000000000000000f >= _e3558_);
                    }
                    let _e3574_ = phi_13414_;
                    let _e3575_ = select(_e3558_, -1000000000000000000f, _e3574_);
                    let _e3576_ = (_e3560_ != _e3560_);
                    if _e3576_ {
                        phi_13429_ = true;
                    } else {
                        phi_13429_ = (-1000000000000000000f >= _e3560_);
                    }
                    let _e3579_ = phi_13429_;
                    let _e3580_ = select(_e3560_, -1000000000000000000f, _e3579_);
                    if (_e3565_ != _e3565_) {
                        phi_13444_ = true;
                    } else {
                        phi_13444_ = (1000000000000000000f <= _e3565_);
                    }
                    let _e3584_ = phi_13444_;
                    let _e3585_ = select(_e3565_, 1000000000000000000f, _e3584_);
                    if (_e3570_ != _e3570_) {
                        phi_13459_ = true;
                    } else {
                        phi_13459_ = (1000000000000000000f <= _e3570_);
                    }
                    let _e3589_ = phi_13459_;
                    let _e3590_ = select(_e3570_, 1000000000000000000f, _e3589_);
                    if (_e3575_ != _e3575_) {
                        phi_13474_ = true;
                    } else {
                        phi_13474_ = (1000000000000000000f <= _e3575_);
                    }
                    let _e3594_ = phi_13474_;
                    let _e3595_ = select(_e3575_, 1000000000000000000f, _e3594_);
                    if (_e3580_ != _e3580_) {
                        phi_13489_ = true;
                    } else {
                        phi_13489_ = (1000000000000000000f <= _e3580_);
                    }
                    let _e3599_ = phi_13489_;
                    let _e3600_ = select(_e3580_, 1000000000000000000f, _e3599_);
                    let _e3614_ = select(0f, _e2830_.x, (bitcast<f32>((bitcast<u32>(_e2830_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3616_ = select(0f, _e2830_.y, (bitcast<f32>((bitcast<u32>(_e2830_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3618_ = select(0f, _e2830_.z, (bitcast<f32>((bitcast<u32>(_e2830_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3620_ = select(0f, _e2830_.w, (bitcast<f32>((bitcast<u32>(_e2830_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3621_ = (_e3614_ != _e3614_);
                    if _e3621_ {
                        phi_13504_ = true;
                    } else {
                        phi_13504_ = (-1000000000000000000f >= _e3614_);
                    }
                    let _e3624_ = phi_13504_;
                    let _e3625_ = select(_e3614_, -1000000000000000000f, _e3624_);
                    let _e3626_ = (_e3616_ != _e3616_);
                    if _e3626_ {
                        phi_13519_ = true;
                    } else {
                        phi_13519_ = (-1000000000000000000f >= _e3616_);
                    }
                    let _e3629_ = phi_13519_;
                    let _e3630_ = select(_e3616_, -1000000000000000000f, _e3629_);
                    let _e3631_ = (_e3618_ != _e3618_);
                    if _e3631_ {
                        phi_13534_ = true;
                    } else {
                        phi_13534_ = (-1000000000000000000f >= _e3618_);
                    }
                    let _e3634_ = phi_13534_;
                    let _e3635_ = select(_e3618_, -1000000000000000000f, _e3634_);
                    let _e3636_ = (_e3620_ != _e3620_);
                    if _e3636_ {
                        phi_13549_ = true;
                    } else {
                        phi_13549_ = (-1000000000000000000f >= _e3620_);
                    }
                    let _e3639_ = phi_13549_;
                    let _e3640_ = select(_e3620_, -1000000000000000000f, _e3639_);
                    if (_e3625_ != _e3625_) {
                        phi_13564_ = true;
                    } else {
                        phi_13564_ = (1000000000000000000f <= _e3625_);
                    }
                    let _e3644_ = phi_13564_;
                    let _e3645_ = select(_e3625_, 1000000000000000000f, _e3644_);
                    if (_e3630_ != _e3630_) {
                        phi_13579_ = true;
                    } else {
                        phi_13579_ = (1000000000000000000f <= _e3630_);
                    }
                    let _e3649_ = phi_13579_;
                    let _e3650_ = select(_e3630_, 1000000000000000000f, _e3649_);
                    if (_e3635_ != _e3635_) {
                        phi_13594_ = true;
                    } else {
                        phi_13594_ = (1000000000000000000f <= _e3635_);
                    }
                    let _e3654_ = phi_13594_;
                    let _e3655_ = select(_e3635_, 1000000000000000000f, _e3654_);
                    if (_e3640_ != _e3640_) {
                        phi_13609_ = true;
                    } else {
                        phi_13609_ = (1000000000000000000f <= _e3640_);
                    }
                    let _e3659_ = phi_13609_;
                    let _e3660_ = select(_e3640_, 1000000000000000000f, _e3659_);
                    let _e3661_ = array<f32, 4>(_e3645_, _e3650_, _e3655_, _e3660_);
                    let _e3662_ = (_e3585_ != _e3585_);
                    if _e3662_ {
                        phi_13624_ = true;
                    } else {
                        phi_13624_ = (_e3645_ <= _e3585_);
                    }
                    let _e3665_ = phi_13624_;
                    let _e3666_ = select(_e3585_, _e3645_, _e3665_);
                    let _e3667_ = (_e3590_ != _e3590_);
                    if _e3667_ {
                        phi_13639_ = true;
                    } else {
                        phi_13639_ = (_e3650_ <= _e3590_);
                    }
                    let _e3670_ = phi_13639_;
                    let _e3671_ = select(_e3590_, _e3650_, _e3670_);
                    let _e3672_ = (_e3595_ != _e3595_);
                    if _e3672_ {
                        phi_13654_ = true;
                    } else {
                        phi_13654_ = (_e3655_ <= _e3595_);
                    }
                    let _e3675_ = phi_13654_;
                    let _e3676_ = select(_e3595_, _e3655_, _e3675_);
                    let _e3677_ = (_e3600_ != _e3600_);
                    if _e3677_ {
                        phi_13669_ = true;
                    } else {
                        phi_13669_ = (_e3660_ <= _e3600_);
                    }
                    let _e3680_ = phi_13669_;
                    let _e3681_ = select(_e3600_, _e3660_, _e3680_);
                    if _e3662_ {
                        phi_13684_ = true;
                    } else {
                        phi_13684_ = (_e3645_ >= _e3585_);
                    }
                    let _e3684_ = phi_13684_;
                    let _e3685_ = select(_e3585_, _e3645_, _e3684_);
                    if _e3667_ {
                        phi_13699_ = true;
                    } else {
                        phi_13699_ = (_e3650_ >= _e3590_);
                    }
                    let _e3688_ = phi_13699_;
                    let _e3689_ = select(_e3590_, _e3650_, _e3688_);
                    if _e3672_ {
                        phi_13714_ = true;
                    } else {
                        phi_13714_ = (_e3655_ >= _e3595_);
                    }
                    let _e3692_ = phi_13714_;
                    let _e3693_ = select(_e3595_, _e3655_, _e3692_);
                    if _e3677_ {
                        phi_13729_ = true;
                    } else {
                        phi_13729_ = (_e3660_ >= _e3600_);
                    }
                    let _e3696_ = phi_13729_;
                    let _e3697_ = select(_e3600_, _e3660_, _e3696_);
                    let _e3698_ = (_e3666_ != _e3666_);
                    if _e3698_ {
                        phi_13744_ = true;
                    } else {
                        phi_13744_ = (-1f >= _e3666_);
                    }
                    let _e3701_ = phi_13744_;
                    let _e3702_ = select(_e3666_, -1f, _e3701_);
                    let _e3703_ = (_e3671_ != _e3671_);
                    if _e3703_ {
                        phi_13759_ = true;
                    } else {
                        phi_13759_ = (-1f >= _e3671_);
                    }
                    let _e3706_ = phi_13759_;
                    let _e3707_ = select(_e3671_, -1f, _e3706_);
                    let _e3708_ = (_e3676_ != _e3676_);
                    if _e3708_ {
                        phi_13774_ = true;
                    } else {
                        phi_13774_ = (-1f >= _e3676_);
                    }
                    let _e3711_ = phi_13774_;
                    let _e3712_ = select(_e3676_, -1f, _e3711_);
                    let _e3713_ = (_e3681_ != _e3681_);
                    if _e3713_ {
                        phi_13789_ = true;
                    } else {
                        phi_13789_ = (-1f >= _e3681_);
                    }
                    let _e3716_ = phi_13789_;
                    let _e3717_ = select(_e3681_, -1f, _e3716_);
                    if (_e3702_ != _e3702_) {
                        phi_13804_ = true;
                    } else {
                        phi_13804_ = (1f <= _e3702_);
                    }
                    let _e3721_ = phi_13804_;
                    let _e3722_ = select(_e3702_, 1f, _e3721_);
                    if (_e3707_ != _e3707_) {
                        phi_13819_ = true;
                    } else {
                        phi_13819_ = (1f <= _e3707_);
                    }
                    let _e3726_ = phi_13819_;
                    let _e3727_ = select(_e3707_, 1f, _e3726_);
                    if (_e3712_ != _e3712_) {
                        phi_13834_ = true;
                    } else {
                        phi_13834_ = (1f <= _e3712_);
                    }
                    let _e3731_ = phi_13834_;
                    let _e3732_ = select(_e3712_, 1f, _e3731_);
                    if (_e3717_ != _e3717_) {
                        phi_13849_ = true;
                    } else {
                        phi_13849_ = (1f <= _e3717_);
                    }
                    let _e3736_ = phi_13849_;
                    let _e3737_ = select(_e3717_, 1f, _e3736_);
                    let _e3738_ = (_e3685_ != _e3685_);
                    if _e3738_ {
                        phi_13864_ = true;
                    } else {
                        phi_13864_ = (-1f >= _e3685_);
                    }
                    let _e3741_ = phi_13864_;
                    let _e3742_ = select(_e3685_, -1f, _e3741_);
                    let _e3743_ = (_e3689_ != _e3689_);
                    if _e3743_ {
                        phi_13879_ = true;
                    } else {
                        phi_13879_ = (-1f >= _e3689_);
                    }
                    let _e3746_ = phi_13879_;
                    let _e3747_ = select(_e3689_, -1f, _e3746_);
                    let _e3748_ = (_e3693_ != _e3693_);
                    if _e3748_ {
                        phi_13894_ = true;
                    } else {
                        phi_13894_ = (-1f >= _e3693_);
                    }
                    let _e3751_ = phi_13894_;
                    let _e3752_ = select(_e3693_, -1f, _e3751_);
                    let _e3753_ = (_e3697_ != _e3697_);
                    if _e3753_ {
                        phi_13909_ = true;
                    } else {
                        phi_13909_ = (-1f >= _e3697_);
                    }
                    let _e3756_ = phi_13909_;
                    let _e3757_ = select(_e3697_, -1f, _e3756_);
                    if (_e3742_ != _e3742_) {
                        phi_13924_ = true;
                    } else {
                        phi_13924_ = (1f <= _e3742_);
                    }
                    let _e3761_ = phi_13924_;
                    let _e3762_ = select(_e3742_, 1f, _e3761_);
                    if (_e3747_ != _e3747_) {
                        phi_13939_ = true;
                    } else {
                        phi_13939_ = (1f <= _e3747_);
                    }
                    let _e3766_ = phi_13939_;
                    let _e3767_ = select(_e3747_, 1f, _e3766_);
                    if (_e3752_ != _e3752_) {
                        phi_13954_ = true;
                    } else {
                        phi_13954_ = (1f <= _e3752_);
                    }
                    let _e3771_ = phi_13954_;
                    let _e3772_ = select(_e3752_, 1f, _e3771_);
                    if (_e3757_ != _e3757_) {
                        phi_13969_ = true;
                    } else {
                        phi_13969_ = (1f <= _e3757_);
                    }
                    let _e3776_ = phi_13969_;
                    let _e3777_ = select(_e3757_, 1f, _e3776_);
                    let _e3782_ = (0.5f * (_e3762_ + _e3722_));
                    let _e3783_ = (0.5f * (_e3767_ + _e3727_));
                    let _e3784_ = (0.5f * (_e3772_ + _e3732_));
                    let _e3785_ = (0.5f * (_e3777_ + _e3737_));
                    let _e3786_ = (_e3762_ - _e3722_);
                    let _e3787_ = (_e3767_ - _e3727_);
                    let _e3788_ = (_e3772_ - _e3732_);
                    let _e3789_ = (_e3777_ - _e3737_);
                    let _e3790_ = (0.5f * _e3786_);
                    let _e3791_ = (0.5f * _e3787_);
                    let _e3792_ = (0.5f * _e3788_);
                    let _e3793_ = (0.5f * _e3789_);
                    let _e3794_ = (_e3782_ * _e3782_);
                    let _e3795_ = (_e3783_ * _e3783_);
                    let _e3796_ = (_e3784_ * _e3784_);
                    let _e3797_ = (_e3785_ * _e3785_);
                    let _e3798_ = (_e3790_ * _e3790_);
                    let _e3799_ = (_e3791_ * _e3791_);
                    let _e3800_ = (_e3792_ * _e3792_);
                    let _e3801_ = (_e3793_ * _e3793_);
                    if (_e3786_ != _e3786_) {
                        phi_13984_ = true;
                    } else {
                        phi_13984_ = (0f >= _e3786_);
                    }
                    let _e3805_ = phi_13984_;
                    let _e3806_ = select(_e3786_, 0f, _e3805_);
                    if (_e3787_ != _e3787_) {
                        phi_13999_ = true;
                    } else {
                        phi_13999_ = (0f >= _e3787_);
                    }
                    let _e3810_ = phi_13999_;
                    let _e3811_ = select(_e3787_, 0f, _e3810_);
                    if (_e3788_ != _e3788_) {
                        phi_14014_ = true;
                    } else {
                        phi_14014_ = (0f >= _e3788_);
                    }
                    let _e3815_ = phi_14014_;
                    let _e3816_ = select(_e3788_, 0f, _e3815_);
                    if (_e3789_ != _e3789_) {
                        phi_14029_ = true;
                    } else {
                        phi_14029_ = (0f >= _e3789_);
                    }
                    let _e3820_ = phi_14029_;
                    let _e3821_ = select(_e3789_, 0f, _e3820_);
                    let _e3822_ = (_e3685_ - _e3666_);
                    let _e3823_ = (_e3689_ - _e3671_);
                    let _e3824_ = (_e3693_ - _e3676_);
                    let _e3825_ = (_e3697_ - _e3681_);
                    if _e3738_ {
                        phi_14044_ = true;
                    } else {
                        phi_14044_ = (-1f <= _e3685_);
                    }
                    let _e3828_ = phi_14044_;
                    if _e3743_ {
                        phi_14059_ = true;
                    } else {
                        phi_14059_ = (-1f <= _e3689_);
                    }
                    let _e3832_ = phi_14059_;
                    if _e3748_ {
                        phi_14074_ = true;
                    } else {
                        phi_14074_ = (-1f <= _e3693_);
                    }
                    let _e3836_ = phi_14074_;
                    if _e3753_ {
                        phi_14089_ = true;
                    } else {
                        phi_14089_ = (-1f <= _e3697_);
                    }
                    let _e3840_ = phi_14089_;
                    let _e3842_ = (select(_e3685_, -1f, _e3828_) - _e3666_);
                    let _e3843_ = (select(_e3689_, -1f, _e3832_) - _e3671_);
                    let _e3844_ = (select(_e3693_, -1f, _e3836_) - _e3676_);
                    let _e3845_ = (select(_e3697_, -1f, _e3840_) - _e3681_);
                    if (_e3842_ != _e3842_) {
                        phi_14104_ = true;
                    } else {
                        phi_14104_ = (0f >= _e3842_);
                    }
                    let _e3849_ = phi_14104_;
                    if (_e3843_ != _e3843_) {
                        phi_14119_ = true;
                    } else {
                        phi_14119_ = (0f >= _e3843_);
                    }
                    let _e3854_ = phi_14119_;
                    if (_e3844_ != _e3844_) {
                        phi_14134_ = true;
                    } else {
                        phi_14134_ = (0f >= _e3844_);
                    }
                    let _e3859_ = phi_14134_;
                    if (_e3845_ != _e3845_) {
                        phi_14149_ = true;
                    } else {
                        phi_14149_ = (0f >= _e3845_);
                    }
                    let _e3864_ = phi_14149_;
                    if _e3698_ {
                        phi_14164_ = true;
                    } else {
                        phi_14164_ = (1f >= _e3666_);
                    }
                    let _e3868_ = phi_14164_;
                    if _e3703_ {
                        phi_14179_ = true;
                    } else {
                        phi_14179_ = (1f >= _e3671_);
                    }
                    let _e3872_ = phi_14179_;
                    if _e3708_ {
                        phi_14194_ = true;
                    } else {
                        phi_14194_ = (1f >= _e3676_);
                    }
                    let _e3876_ = phi_14194_;
                    if _e3713_ {
                        phi_14209_ = true;
                    } else {
                        phi_14209_ = (1f >= _e3681_);
                    }
                    let _e3880_ = phi_14209_;
                    let _e3882_ = (_e3685_ - select(_e3666_, 1f, _e3868_));
                    let _e3883_ = (_e3689_ - select(_e3671_, 1f, _e3872_));
                    let _e3884_ = (_e3693_ - select(_e3676_, 1f, _e3876_));
                    let _e3885_ = (_e3697_ - select(_e3681_, 1f, _e3880_));
                    if (_e3882_ != _e3882_) {
                        phi_14224_ = true;
                    } else {
                        phi_14224_ = (0f >= _e3882_);
                    }
                    let _e3889_ = phi_14224_;
                    if (_e3883_ != _e3883_) {
                        phi_14239_ = true;
                    } else {
                        phi_14239_ = (0f >= _e3883_);
                    }
                    let _e3894_ = phi_14239_;
                    if (_e3884_ != _e3884_) {
                        phi_14254_ = true;
                    } else {
                        phi_14254_ = (0f >= _e3884_);
                    }
                    let _e3899_ = phi_14254_;
                    if (_e3885_ != _e3885_) {
                        phi_14269_ = true;
                    } else {
                        phi_14269_ = (0f >= _e3885_);
                    }
                    let _e3904_ = phi_14269_;
                    let _e3908_ = bitcast<f32>((bitcast<u32>(_e3585_) & 2147483647u));
                    let _e3911_ = bitcast<f32>((bitcast<u32>(_e3590_) & 2147483647u));
                    let _e3914_ = bitcast<f32>((bitcast<u32>(_e3595_) & 2147483647u));
                    let _e3917_ = bitcast<f32>((bitcast<u32>(_e3600_) & 2147483647u));
                    let _e3920_ = bitcast<f32>((bitcast<u32>(_e3645_) & 2147483647u));
                    let _e3923_ = bitcast<f32>((bitcast<u32>(_e3650_) & 2147483647u));
                    let _e3926_ = bitcast<f32>((bitcast<u32>(_e3655_) & 2147483647u));
                    let _e3929_ = bitcast<f32>((bitcast<u32>(_e3660_) & 2147483647u));
                    if (_e3908_ != _e3908_) {
                        phi_14284_ = true;
                    } else {
                        phi_14284_ = (_e3920_ >= _e3908_);
                    }
                    let _e3933_ = phi_14284_;
                    let _e3934_ = select(_e3908_, _e3920_, _e3933_);
                    if (_e3911_ != _e3911_) {
                        phi_14299_ = true;
                    } else {
                        phi_14299_ = (_e3923_ >= _e3911_);
                    }
                    let _e3938_ = phi_14299_;
                    let _e3939_ = select(_e3911_, _e3923_, _e3938_);
                    if (_e3914_ != _e3914_) {
                        phi_14314_ = true;
                    } else {
                        phi_14314_ = (_e3926_ >= _e3914_);
                    }
                    let _e3943_ = phi_14314_;
                    let _e3944_ = select(_e3914_, _e3926_, _e3943_);
                    if (_e3917_ != _e3917_) {
                        phi_14329_ = true;
                    } else {
                        phi_14329_ = (_e3929_ >= _e3917_);
                    }
                    let _e3948_ = phi_14329_;
                    let _e3949_ = select(_e3917_, _e3929_, _e3948_);
                    let _e3966_ = (_e3934_ <= 0f);
                    let _e3967_ = select((_e3782_ * (1.5f - (0.5f * (_e3794_ + _e3798_)))), 0f, _e3966_);
                    let _e3968_ = (_e3939_ <= 0f);
                    let _e3969_ = select((_e3783_ * (1.5f - (0.5f * (_e3795_ + _e3799_)))), 0f, _e3968_);
                    let _e3970_ = (_e3944_ <= 0f);
                    let _e3971_ = select((_e3784_ * (1.5f - (0.5f * (_e3796_ + _e3800_)))), 0f, _e3970_);
                    let _e3972_ = (_e3949_ <= 0f);
                    let _e3973_ = select((_e3785_ * (1.5f - (0.5f * (_e3797_ + _e3801_)))), 0f, _e3972_);
                    let _e4015_ = select(((_e3794_ + (0.33333334f * _e3798_)) - (((_e3794_ * _e3794_) + ((2f * _e3794_) * _e3798_)) + (0.2f * (_e3798_ * _e3798_)))), 0f, _e3966_);
                    let _e4016_ = select(((_e3795_ + (0.33333334f * _e3799_)) - (((_e3795_ * _e3795_) + ((2f * _e3795_) * _e3799_)) + (0.2f * (_e3799_ * _e3799_)))), 0f, _e3968_);
                    let _e4017_ = select(((_e3796_ + (0.33333334f * _e3800_)) - (((_e3796_ * _e3796_) + ((2f * _e3796_) * _e3800_)) + (0.2f * (_e3800_ * _e3800_)))), 0f, _e3970_);
                    let _e4018_ = select(((_e3797_ + (0.33333334f * _e3801_)) - (((_e3797_ * _e3797_) + ((2f * _e3797_) * _e3801_)) + (0.2f * (_e3801_ * _e3801_)))), 0f, _e3972_);
                    if (_e3934_ <= 1f) {
                        if (_e3939_ <= 1f) {
                            let _e4022_ = (_e3944_ <= 1f);
                            if _e4022_ {
                                phi_7925_ = (_e3949_ <= 1f);
                            } else {
                                phi_7925_ = bool();
                            }
                            let _e4025_ = phi_7925_;
                            phi_7927_ = _e4025_;
                            phi_7928_ = select(true, false, _e4022_);
                        } else {
                            phi_7927_ = bool();
                            phi_7928_ = true;
                        }
                        let _e4028_ = phi_7927_;
                        let _e4030_ = phi_7928_;
                        phi_7929_ = _e4028_;
                        phi_7930_ = _e4030_;
                    } else {
                        phi_7929_ = bool();
                        phi_7930_ = true;
                    }
                    let _e4032_ = phi_7929_;
                    let _e4034_ = phi_7930_;
                    if select(_e4032_, false, _e4034_) {
                        phi_8308_ = type_33_(type_24_(array<f32, 4>(_e3967_, _e3969_, _e3971_, _e3973_)), type_24_(array<f32, 4>(_e4015_, _e4016_, _e4017_, _e4018_)), type_24_(_e3661_));
                    } else {
                        if (_e3822_ != _e3822_) {
                            phi_14344_ = true;
                        } else {
                            phi_14344_ = (0.000001f >= _e3822_);
                        }
                        let _e4039_ = phi_14344_;
                        let _e4040_ = select(_e3822_, 0.000001f, _e4039_);
                        if (_e3823_ != _e3823_) {
                            phi_14359_ = true;
                        } else {
                            phi_14359_ = (0.000001f >= _e3823_);
                        }
                        let _e4044_ = phi_14359_;
                        let _e4045_ = select(_e3823_, 0.000001f, _e4044_);
                        if (_e3824_ != _e3824_) {
                            phi_14374_ = true;
                        } else {
                            phi_14374_ = (0.000001f >= _e3824_);
                        }
                        let _e4049_ = phi_14374_;
                        let _e4050_ = select(_e3824_, 0.000001f, _e4049_);
                        if (_e3825_ != _e3825_) {
                            phi_14389_ = true;
                        } else {
                            phi_14389_ = (0.000001f >= _e3825_);
                        }
                        let _e4054_ = phi_14389_;
                        let _e4055_ = select(_e3825_, 0.000001f, _e4054_);
                        let _e4058_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4040_)));
                        let _e4061_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4045_)));
                        let _e4064_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4050_)));
                        let _e4067_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4055_)));
                        let _e4077_ = (_e4058_ + (_e4058_ * (1f - (_e4040_ * _e4058_))));
                        let _e4079_ = (_e4061_ + (_e4061_ * (1f - (_e4045_ * _e4061_))));
                        let _e4081_ = (_e4064_ + (_e4064_ * (1f - (_e4050_ * _e4064_))));
                        let _e4083_ = (_e4067_ + (_e4067_ * (1f - (_e4055_ * _e4067_))));
                        let _e4093_ = (_e4077_ + (_e4077_ * (1f - (_e4040_ * _e4077_))));
                        let _e4095_ = (_e4079_ + (_e4079_ * (1f - (_e4045_ * _e4079_))));
                        let _e4097_ = (_e4081_ + (_e4081_ * (1f - (_e4050_ * _e4081_))));
                        let _e4099_ = (_e4083_ + (_e4083_ * (1f - (_e4055_ * _e4083_))));
                        let _e4152_ = (select((select(0f, 1f, (-1f <= _e3666_)) * select(0f, 1f, (_e3685_ <= 1f))), 1f, (_e3822_ <= 0.000001f)) <= 0f);
                        let _e4154_ = (select((select(0f, 1f, (-1f <= _e3671_)) * select(0f, 1f, (_e3689_ <= 1f))), 1f, (_e3823_ <= 0.000001f)) <= 0f);
                        let _e4156_ = (select((select(0f, 1f, (-1f <= _e3676_)) * select(0f, 1f, (_e3693_ <= 1f))), 1f, (_e3824_ <= 0.000001f)) <= 0f);
                        let _e4158_ = (select((select(0f, 1f, (-1f <= _e3681_)) * select(0f, 1f, (_e3697_ <= 1f))), 1f, (_e3825_ <= 0.000001f)) <= 0f);
                        let _e4164_ = (1f <= _e3666_);
                        let _e4166_ = (1f <= _e3671_);
                        let _e4168_ = (1f <= _e3676_);
                        let _e4170_ = (1f <= _e3681_);
                        let _e4172_ = (_e3685_ <= -1f);
                        let _e4174_ = (_e3689_ <= -1f);
                        let _e4176_ = (_e3693_ <= -1f);
                        let _e4178_ = (_e3697_ <= -1f);
                        phi_8308_ = type_33_(type_24_(array<f32, 4>(select(select(select(_e3967_, (((select(_e3882_, 0f, _e3889_) - select(_e3842_, 0f, _e3849_)) + (_e3806_ * _e3967_)) * _e4093_), _e4152_), 1f, _e4164_), -1f, _e4172_), select(select(select(_e3969_, (((select(_e3883_, 0f, _e3894_) - select(_e3843_, 0f, _e3854_)) + (_e3811_ * _e3969_)) * _e4095_), _e4154_), 1f, _e4166_), -1f, _e4174_), select(select(select(_e3971_, (((select(_e3884_, 0f, _e3899_) - select(_e3844_, 0f, _e3859_)) + (_e3816_ * _e3971_)) * _e4097_), _e4156_), 1f, _e4168_), -1f, _e4176_), select(select(select(_e3973_, (((select(_e3885_, 0f, _e3904_) - select(_e3845_, 0f, _e3864_)) + (_e3821_ * _e3973_)) * _e4099_), _e4158_), 1f, _e4170_), -1f, _e4178_))), type_24_(array<f32, 4>(select(select(select(_e4015_, ((_e3806_ * _e4015_) * _e4093_), _e4152_), 0f, _e4164_), 0f, _e4172_), select(select(select(_e4016_, ((_e3811_ * _e4016_) * _e4095_), _e4154_), 0f, _e4166_), 0f, _e4174_), select(select(select(_e4017_, ((_e3816_ * _e4017_) * _e4097_), _e4156_), 0f, _e4168_), 0f, _e4176_), select(select(select(_e4018_, ((_e3821_ * _e4018_) * _e4099_), _e4158_), 0f, _e4170_), 0f, _e4178_))), type_24_(_e3661_));
                    }
                    let _e4199_ = phi_8308_;
                    if (_e400_ <= 0f) {
                        if (_e420_ <= 0f) {
                            let _e4203_ = (_e440_ <= 0f);
                            if _e4203_ {
                                phi_8325_ = (_e460_ <= 0f);
                            } else {
                                phi_8325_ = bool();
                            }
                            let _e4206_ = phi_8325_;
                            phi_8327_ = _e4206_;
                            phi_8328_ = select(true, false, _e4203_);
                        } else {
                            phi_8327_ = bool();
                            phi_8328_ = true;
                        }
                        let _e4209_ = phi_8327_;
                        let _e4211_ = phi_8328_;
                        phi_8329_ = _e4209_;
                        phi_8330_ = _e4211_;
                    } else {
                        phi_8329_ = bool();
                        phi_8330_ = true;
                    }
                    let _e4213_ = phi_8329_;
                    let _e4215_ = phi_8330_;
                    if select(_e4213_, false, _e4215_) {
                        phi_10218_ = _e4199_.member;
                        phi_10219_ = _e3531_.member;
                    } else {
                        if _e2893_ {
                            phi_14404_ = true;
                        } else {
                            phi_14404_ = (-1000000000000000000f >= _e2886_);
                        }
                        let _e4219_ = phi_14404_;
                        let _e4220_ = select(_e2886_, -1000000000000000000f, _e4219_);
                        if _e2898_ {
                            phi_14419_ = true;
                        } else {
                            phi_14419_ = (-1000000000000000000f >= _e2888_);
                        }
                        let _e4223_ = phi_14419_;
                        let _e4224_ = select(_e2888_, -1000000000000000000f, _e4223_);
                        if _e2903_ {
                            phi_14434_ = true;
                        } else {
                            phi_14434_ = (-1000000000000000000f >= _e2890_);
                        }
                        let _e4227_ = phi_14434_;
                        let _e4228_ = select(_e2890_, -1000000000000000000f, _e4227_);
                        if _e2908_ {
                            phi_14449_ = true;
                        } else {
                            phi_14449_ = (-1000000000000000000f >= _e2892_);
                        }
                        let _e4231_ = phi_14449_;
                        let _e4232_ = select(_e2892_, -1000000000000000000f, _e4231_);
                        if (_e4220_ != _e4220_) {
                            phi_14464_ = true;
                        } else {
                            phi_14464_ = (1000000000000000000f <= _e4220_);
                        }
                        let _e4236_ = phi_14464_;
                        let _e4237_ = select(_e4220_, 1000000000000000000f, _e4236_);
                        if (_e4224_ != _e4224_) {
                            phi_14479_ = true;
                        } else {
                            phi_14479_ = (1000000000000000000f <= _e4224_);
                        }
                        let _e4241_ = phi_14479_;
                        let _e4242_ = select(_e4224_, 1000000000000000000f, _e4241_);
                        if (_e4228_ != _e4228_) {
                            phi_14494_ = true;
                        } else {
                            phi_14494_ = (1000000000000000000f <= _e4228_);
                        }
                        let _e4246_ = phi_14494_;
                        let _e4247_ = select(_e4228_, 1000000000000000000f, _e4246_);
                        if (_e4232_ != _e4232_) {
                            phi_14509_ = true;
                        } else {
                            phi_14509_ = (1000000000000000000f <= _e4232_);
                        }
                        let _e4251_ = phi_14509_;
                        let _e4252_ = select(_e4232_, 1000000000000000000f, _e4251_);
                        if _e2953_ {
                            phi_14524_ = true;
                        } else {
                            phi_14524_ = (-1000000000000000000f >= _e2946_);
                        }
                        let _e4255_ = phi_14524_;
                        let _e4256_ = select(_e2946_, -1000000000000000000f, _e4255_);
                        if _e2958_ {
                            phi_14539_ = true;
                        } else {
                            phi_14539_ = (-1000000000000000000f >= _e2948_);
                        }
                        let _e4259_ = phi_14539_;
                        let _e4260_ = select(_e2948_, -1000000000000000000f, _e4259_);
                        if _e2963_ {
                            phi_14554_ = true;
                        } else {
                            phi_14554_ = (-1000000000000000000f >= _e2950_);
                        }
                        let _e4263_ = phi_14554_;
                        let _e4264_ = select(_e2950_, -1000000000000000000f, _e4263_);
                        if _e2968_ {
                            phi_14569_ = true;
                        } else {
                            phi_14569_ = (-1000000000000000000f >= _e2952_);
                        }
                        let _e4267_ = phi_14569_;
                        let _e4268_ = select(_e2952_, -1000000000000000000f, _e4267_);
                        if (_e4256_ != _e4256_) {
                            phi_14584_ = true;
                        } else {
                            phi_14584_ = (1000000000000000000f <= _e4256_);
                        }
                        let _e4272_ = phi_14584_;
                        let _e4273_ = select(_e4256_, 1000000000000000000f, _e4272_);
                        if (_e4260_ != _e4260_) {
                            phi_14599_ = true;
                        } else {
                            phi_14599_ = (1000000000000000000f <= _e4260_);
                        }
                        let _e4277_ = phi_14599_;
                        let _e4278_ = select(_e4260_, 1000000000000000000f, _e4277_);
                        if (_e4264_ != _e4264_) {
                            phi_14614_ = true;
                        } else {
                            phi_14614_ = (1000000000000000000f <= _e4264_);
                        }
                        let _e4282_ = phi_14614_;
                        let _e4283_ = select(_e4264_, 1000000000000000000f, _e4282_);
                        if (_e4268_ != _e4268_) {
                            phi_14629_ = true;
                        } else {
                            phi_14629_ = (1000000000000000000f <= _e4268_);
                        }
                        let _e4287_ = phi_14629_;
                        let _e4288_ = select(_e4268_, 1000000000000000000f, _e4287_);
                        let _e4289_ = (_e4237_ != _e4237_);
                        if _e4289_ {
                            phi_14644_ = true;
                        } else {
                            phi_14644_ = (_e4273_ <= _e4237_);
                        }
                        let _e4292_ = phi_14644_;
                        let _e4293_ = select(_e4237_, _e4273_, _e4292_);
                        let _e4294_ = (_e4242_ != _e4242_);
                        if _e4294_ {
                            phi_14659_ = true;
                        } else {
                            phi_14659_ = (_e4278_ <= _e4242_);
                        }
                        let _e4297_ = phi_14659_;
                        let _e4298_ = select(_e4242_, _e4278_, _e4297_);
                        let _e4299_ = (_e4247_ != _e4247_);
                        if _e4299_ {
                            phi_14674_ = true;
                        } else {
                            phi_14674_ = (_e4283_ <= _e4247_);
                        }
                        let _e4302_ = phi_14674_;
                        let _e4303_ = select(_e4247_, _e4283_, _e4302_);
                        let _e4304_ = (_e4252_ != _e4252_);
                        if _e4304_ {
                            phi_14689_ = true;
                        } else {
                            phi_14689_ = (_e4288_ <= _e4252_);
                        }
                        let _e4307_ = phi_14689_;
                        let _e4308_ = select(_e4252_, _e4288_, _e4307_);
                        if _e4289_ {
                            phi_14704_ = true;
                        } else {
                            phi_14704_ = (_e4273_ >= _e4237_);
                        }
                        let _e4311_ = phi_14704_;
                        let _e4312_ = select(_e4237_, _e4273_, _e4311_);
                        if _e4294_ {
                            phi_14719_ = true;
                        } else {
                            phi_14719_ = (_e4278_ >= _e4242_);
                        }
                        let _e4315_ = phi_14719_;
                        let _e4316_ = select(_e4242_, _e4278_, _e4315_);
                        if _e4299_ {
                            phi_14734_ = true;
                        } else {
                            phi_14734_ = (_e4283_ >= _e4247_);
                        }
                        let _e4319_ = phi_14734_;
                        let _e4320_ = select(_e4247_, _e4283_, _e4319_);
                        if _e4304_ {
                            phi_14749_ = true;
                        } else {
                            phi_14749_ = (_e4288_ >= _e4252_);
                        }
                        let _e4323_ = phi_14749_;
                        let _e4324_ = select(_e4252_, _e4288_, _e4323_);
                        if (_e4293_ != _e4293_) {
                            phi_14764_ = true;
                        } else {
                            phi_14764_ = (-1f >= _e4293_);
                        }
                        let _e4328_ = phi_14764_;
                        let _e4329_ = select(_e4293_, -1f, _e4328_);
                        if (_e4298_ != _e4298_) {
                            phi_14779_ = true;
                        } else {
                            phi_14779_ = (-1f >= _e4298_);
                        }
                        let _e4333_ = phi_14779_;
                        let _e4334_ = select(_e4298_, -1f, _e4333_);
                        if (_e4303_ != _e4303_) {
                            phi_14794_ = true;
                        } else {
                            phi_14794_ = (-1f >= _e4303_);
                        }
                        let _e4338_ = phi_14794_;
                        let _e4339_ = select(_e4303_, -1f, _e4338_);
                        if (_e4308_ != _e4308_) {
                            phi_14809_ = true;
                        } else {
                            phi_14809_ = (-1f >= _e4308_);
                        }
                        let _e4343_ = phi_14809_;
                        let _e4344_ = select(_e4308_, -1f, _e4343_);
                        if (_e4329_ != _e4329_) {
                            phi_14824_ = true;
                        } else {
                            phi_14824_ = (1f <= _e4329_);
                        }
                        let _e4348_ = phi_14824_;
                        let _e4349_ = select(_e4329_, 1f, _e4348_);
                        if (_e4334_ != _e4334_) {
                            phi_14839_ = true;
                        } else {
                            phi_14839_ = (1f <= _e4334_);
                        }
                        let _e4353_ = phi_14839_;
                        let _e4354_ = select(_e4334_, 1f, _e4353_);
                        if (_e4339_ != _e4339_) {
                            phi_14854_ = true;
                        } else {
                            phi_14854_ = (1f <= _e4339_);
                        }
                        let _e4358_ = phi_14854_;
                        let _e4359_ = select(_e4339_, 1f, _e4358_);
                        if (_e4344_ != _e4344_) {
                            phi_14869_ = true;
                        } else {
                            phi_14869_ = (1f <= _e4344_);
                        }
                        let _e4363_ = phi_14869_;
                        let _e4364_ = select(_e4344_, 1f, _e4363_);
                        if (_e4312_ != _e4312_) {
                            phi_14884_ = true;
                        } else {
                            phi_14884_ = (-1f >= _e4312_);
                        }
                        let _e4368_ = phi_14884_;
                        let _e4369_ = select(_e4312_, -1f, _e4368_);
                        if (_e4316_ != _e4316_) {
                            phi_14899_ = true;
                        } else {
                            phi_14899_ = (-1f >= _e4316_);
                        }
                        let _e4373_ = phi_14899_;
                        let _e4374_ = select(_e4316_, -1f, _e4373_);
                        if (_e4320_ != _e4320_) {
                            phi_14914_ = true;
                        } else {
                            phi_14914_ = (-1f >= _e4320_);
                        }
                        let _e4378_ = phi_14914_;
                        let _e4379_ = select(_e4320_, -1f, _e4378_);
                        if (_e4324_ != _e4324_) {
                            phi_14929_ = true;
                        } else {
                            phi_14929_ = (-1f >= _e4324_);
                        }
                        let _e4383_ = phi_14929_;
                        let _e4384_ = select(_e4324_, -1f, _e4383_);
                        if (_e4369_ != _e4369_) {
                            phi_14944_ = true;
                        } else {
                            phi_14944_ = (1f <= _e4369_);
                        }
                        let _e4388_ = phi_14944_;
                        let _e4389_ = select(_e4369_, 1f, _e4388_);
                        if (_e4374_ != _e4374_) {
                            phi_14959_ = true;
                        } else {
                            phi_14959_ = (1f <= _e4374_);
                        }
                        let _e4393_ = phi_14959_;
                        let _e4394_ = select(_e4374_, 1f, _e4393_);
                        if (_e4379_ != _e4379_) {
                            phi_14974_ = true;
                        } else {
                            phi_14974_ = (1f <= _e4379_);
                        }
                        let _e4398_ = phi_14974_;
                        let _e4399_ = select(_e4379_, 1f, _e4398_);
                        if (_e4384_ != _e4384_) {
                            phi_14989_ = true;
                        } else {
                            phi_14989_ = (1f <= _e4384_);
                        }
                        let _e4403_ = phi_14989_;
                        let _e4404_ = select(_e4384_, 1f, _e4403_);
                        let _e4409_ = (0.5f * (_e4389_ + _e4349_));
                        let _e4410_ = (0.5f * (_e4394_ + _e4354_));
                        let _e4411_ = (0.5f * (_e4399_ + _e4359_));
                        let _e4412_ = (0.5f * (_e4404_ + _e4364_));
                        let _e4413_ = (_e4389_ - _e4349_);
                        let _e4414_ = (_e4394_ - _e4354_);
                        let _e4415_ = (_e4399_ - _e4359_);
                        let _e4416_ = (_e4404_ - _e4364_);
                        let _e4417_ = (0.5f * _e4413_);
                        let _e4418_ = (0.5f * _e4414_);
                        let _e4419_ = (0.5f * _e4415_);
                        let _e4420_ = (0.5f * _e4416_);
                        let _e4421_ = (_e4409_ * _e4409_);
                        let _e4422_ = (_e4410_ * _e4410_);
                        let _e4423_ = (_e4411_ * _e4411_);
                        let _e4424_ = (_e4412_ * _e4412_);
                        let _e4425_ = (_e4417_ * _e4417_);
                        let _e4426_ = (_e4418_ * _e4418_);
                        let _e4427_ = (_e4419_ * _e4419_);
                        let _e4428_ = (_e4420_ * _e4420_);
                        if (_e4413_ != _e4413_) {
                            phi_15004_ = true;
                        } else {
                            phi_15004_ = (0f >= _e4413_);
                        }
                        let _e4432_ = phi_15004_;
                        if (_e4414_ != _e4414_) {
                            phi_15019_ = true;
                        } else {
                            phi_15019_ = (0f >= _e4414_);
                        }
                        let _e4437_ = phi_15019_;
                        if (_e4415_ != _e4415_) {
                            phi_15034_ = true;
                        } else {
                            phi_15034_ = (0f >= _e4415_);
                        }
                        let _e4442_ = phi_15034_;
                        if (_e4416_ != _e4416_) {
                            phi_15049_ = true;
                        } else {
                            phi_15049_ = (0f >= _e4416_);
                        }
                        let _e4447_ = phi_15049_;
                        let _e4449_ = (_e4312_ - _e4293_);
                        let _e4450_ = (_e4316_ - _e4298_);
                        let _e4451_ = (_e4320_ - _e4303_);
                        let _e4452_ = (_e4324_ - _e4308_);
                        let _e4455_ = bitcast<f32>((bitcast<u32>(_e4237_) & 2147483647u));
                        let _e4458_ = bitcast<f32>((bitcast<u32>(_e4242_) & 2147483647u));
                        let _e4461_ = bitcast<f32>((bitcast<u32>(_e4247_) & 2147483647u));
                        let _e4464_ = bitcast<f32>((bitcast<u32>(_e4252_) & 2147483647u));
                        let _e4467_ = bitcast<f32>((bitcast<u32>(_e4273_) & 2147483647u));
                        let _e4470_ = bitcast<f32>((bitcast<u32>(_e4278_) & 2147483647u));
                        let _e4473_ = bitcast<f32>((bitcast<u32>(_e4283_) & 2147483647u));
                        let _e4476_ = bitcast<f32>((bitcast<u32>(_e4288_) & 2147483647u));
                        if (_e4455_ != _e4455_) {
                            phi_15304_ = true;
                        } else {
                            phi_15304_ = (_e4467_ >= _e4455_);
                        }
                        let _e4480_ = phi_15304_;
                        if (_e4458_ != _e4458_) {
                            phi_15319_ = true;
                        } else {
                            phi_15319_ = (_e4470_ >= _e4458_);
                        }
                        let _e4485_ = phi_15319_;
                        if (_e4461_ != _e4461_) {
                            phi_15334_ = true;
                        } else {
                            phi_15334_ = (_e4473_ >= _e4461_);
                        }
                        let _e4490_ = phi_15334_;
                        if (_e4464_ != _e4464_) {
                            phi_15349_ = true;
                        } else {
                            phi_15349_ = (_e4476_ >= _e4464_);
                        }
                        let _e4495_ = phi_15349_;
                        let _e4501_ = (1f - (_e4421_ + _e4425_));
                        let _e4502_ = (1f - (_e4422_ + _e4426_));
                        let _e4503_ = (1f - (_e4423_ + _e4427_));
                        let _e4504_ = (1f - (_e4424_ + _e4428_));
                        let _e4530_ = select(((_e4409_ * _e4501_) * ((_e4501_ * _e4501_) + ((4f * _e4421_) * _e4425_))), 0f, (select(_e4455_, _e4467_, _e4480_) <= 0f));
                        let _e4532_ = select(((_e4410_ * _e4502_) * ((_e4502_ * _e4502_) + ((4f * _e4422_) * _e4426_))), 0f, (select(_e4458_, _e4470_, _e4485_) <= 0f));
                        let _e4534_ = select(((_e4411_ * _e4503_) * ((_e4503_ * _e4503_) + ((4f * _e4423_) * _e4427_))), 0f, (select(_e4461_, _e4473_, _e4490_) <= 0f));
                        let _e4536_ = select(((_e4412_ * _e4504_) * ((_e4504_ * _e4504_) + ((4f * _e4424_) * _e4428_))), 0f, (select(_e4464_, _e4476_, _e4495_) <= 0f));
                        if (_e4449_ != _e4449_) {
                            phi_15364_ = true;
                        } else {
                            phi_15364_ = (0.000001f >= _e4449_);
                        }
                        let _e4540_ = phi_15364_;
                        let _e4541_ = select(_e4449_, 0.000001f, _e4540_);
                        if (_e4450_ != _e4450_) {
                            phi_15379_ = true;
                        } else {
                            phi_15379_ = (0.000001f >= _e4450_);
                        }
                        let _e4545_ = phi_15379_;
                        let _e4546_ = select(_e4450_, 0.000001f, _e4545_);
                        if (_e4451_ != _e4451_) {
                            phi_15394_ = true;
                        } else {
                            phi_15394_ = (0.000001f >= _e4451_);
                        }
                        let _e4550_ = phi_15394_;
                        let _e4551_ = select(_e4451_, 0.000001f, _e4550_);
                        if (_e4452_ != _e4452_) {
                            phi_15409_ = true;
                        } else {
                            phi_15409_ = (0.000001f >= _e4452_);
                        }
                        let _e4555_ = phi_15409_;
                        let _e4556_ = select(_e4452_, 0.000001f, _e4555_);
                        let _e4559_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4541_)));
                        let _e4562_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4546_)));
                        let _e4565_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4551_)));
                        let _e4568_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4556_)));
                        let _e4578_ = (_e4559_ + (_e4559_ * (1f - (_e4541_ * _e4559_))));
                        let _e4580_ = (_e4562_ + (_e4562_ * (1f - (_e4546_ * _e4562_))));
                        let _e4582_ = (_e4565_ + (_e4565_ * (1f - (_e4551_ * _e4565_))));
                        let _e4584_ = (_e4568_ + (_e4568_ * (1f - (_e4556_ * _e4568_))));
                        if _e3561_ {
                            phi_15424_ = true;
                        } else {
                            phi_15424_ = (-1000000000000000000f >= _e3554_);
                        }
                        let _e4663_ = phi_15424_;
                        let _e4664_ = select(_e3554_, -1000000000000000000f, _e4663_);
                        if _e3566_ {
                            phi_15439_ = true;
                        } else {
                            phi_15439_ = (-1000000000000000000f >= _e3556_);
                        }
                        let _e4667_ = phi_15439_;
                        let _e4668_ = select(_e3556_, -1000000000000000000f, _e4667_);
                        if _e3571_ {
                            phi_15454_ = true;
                        } else {
                            phi_15454_ = (-1000000000000000000f >= _e3558_);
                        }
                        let _e4671_ = phi_15454_;
                        let _e4672_ = select(_e3558_, -1000000000000000000f, _e4671_);
                        if _e3576_ {
                            phi_15469_ = true;
                        } else {
                            phi_15469_ = (-1000000000000000000f >= _e3560_);
                        }
                        let _e4675_ = phi_15469_;
                        let _e4676_ = select(_e3560_, -1000000000000000000f, _e4675_);
                        if (_e4664_ != _e4664_) {
                            phi_15484_ = true;
                        } else {
                            phi_15484_ = (1000000000000000000f <= _e4664_);
                        }
                        let _e4680_ = phi_15484_;
                        let _e4681_ = select(_e4664_, 1000000000000000000f, _e4680_);
                        if (_e4668_ != _e4668_) {
                            phi_15499_ = true;
                        } else {
                            phi_15499_ = (1000000000000000000f <= _e4668_);
                        }
                        let _e4685_ = phi_15499_;
                        let _e4686_ = select(_e4668_, 1000000000000000000f, _e4685_);
                        if (_e4672_ != _e4672_) {
                            phi_15514_ = true;
                        } else {
                            phi_15514_ = (1000000000000000000f <= _e4672_);
                        }
                        let _e4690_ = phi_15514_;
                        let _e4691_ = select(_e4672_, 1000000000000000000f, _e4690_);
                        if (_e4676_ != _e4676_) {
                            phi_15529_ = true;
                        } else {
                            phi_15529_ = (1000000000000000000f <= _e4676_);
                        }
                        let _e4695_ = phi_15529_;
                        let _e4696_ = select(_e4676_, 1000000000000000000f, _e4695_);
                        if _e3621_ {
                            phi_15544_ = true;
                        } else {
                            phi_15544_ = (-1000000000000000000f >= _e3614_);
                        }
                        let _e4699_ = phi_15544_;
                        let _e4700_ = select(_e3614_, -1000000000000000000f, _e4699_);
                        if _e3626_ {
                            phi_15559_ = true;
                        } else {
                            phi_15559_ = (-1000000000000000000f >= _e3616_);
                        }
                        let _e4703_ = phi_15559_;
                        let _e4704_ = select(_e3616_, -1000000000000000000f, _e4703_);
                        if _e3631_ {
                            phi_15574_ = true;
                        } else {
                            phi_15574_ = (-1000000000000000000f >= _e3618_);
                        }
                        let _e4707_ = phi_15574_;
                        let _e4708_ = select(_e3618_, -1000000000000000000f, _e4707_);
                        if _e3636_ {
                            phi_15589_ = true;
                        } else {
                            phi_15589_ = (-1000000000000000000f >= _e3620_);
                        }
                        let _e4711_ = phi_15589_;
                        let _e4712_ = select(_e3620_, -1000000000000000000f, _e4711_);
                        if (_e4700_ != _e4700_) {
                            phi_15604_ = true;
                        } else {
                            phi_15604_ = (1000000000000000000f <= _e4700_);
                        }
                        let _e4716_ = phi_15604_;
                        let _e4717_ = select(_e4700_, 1000000000000000000f, _e4716_);
                        if (_e4704_ != _e4704_) {
                            phi_15619_ = true;
                        } else {
                            phi_15619_ = (1000000000000000000f <= _e4704_);
                        }
                        let _e4721_ = phi_15619_;
                        let _e4722_ = select(_e4704_, 1000000000000000000f, _e4721_);
                        if (_e4708_ != _e4708_) {
                            phi_15634_ = true;
                        } else {
                            phi_15634_ = (1000000000000000000f <= _e4708_);
                        }
                        let _e4726_ = phi_15634_;
                        let _e4727_ = select(_e4708_, 1000000000000000000f, _e4726_);
                        if (_e4712_ != _e4712_) {
                            phi_15649_ = true;
                        } else {
                            phi_15649_ = (1000000000000000000f <= _e4712_);
                        }
                        let _e4731_ = phi_15649_;
                        let _e4732_ = select(_e4712_, 1000000000000000000f, _e4731_);
                        let _e4733_ = (_e4681_ != _e4681_);
                        if _e4733_ {
                            phi_15664_ = true;
                        } else {
                            phi_15664_ = (_e4717_ <= _e4681_);
                        }
                        let _e4736_ = phi_15664_;
                        let _e4737_ = select(_e4681_, _e4717_, _e4736_);
                        let _e4738_ = (_e4686_ != _e4686_);
                        if _e4738_ {
                            phi_15679_ = true;
                        } else {
                            phi_15679_ = (_e4722_ <= _e4686_);
                        }
                        let _e4741_ = phi_15679_;
                        let _e4742_ = select(_e4686_, _e4722_, _e4741_);
                        let _e4743_ = (_e4691_ != _e4691_);
                        if _e4743_ {
                            phi_15694_ = true;
                        } else {
                            phi_15694_ = (_e4727_ <= _e4691_);
                        }
                        let _e4746_ = phi_15694_;
                        let _e4747_ = select(_e4691_, _e4727_, _e4746_);
                        let _e4748_ = (_e4696_ != _e4696_);
                        if _e4748_ {
                            phi_15709_ = true;
                        } else {
                            phi_15709_ = (_e4732_ <= _e4696_);
                        }
                        let _e4751_ = phi_15709_;
                        let _e4752_ = select(_e4696_, _e4732_, _e4751_);
                        if _e4733_ {
                            phi_15724_ = true;
                        } else {
                            phi_15724_ = (_e4717_ >= _e4681_);
                        }
                        let _e4755_ = phi_15724_;
                        let _e4756_ = select(_e4681_, _e4717_, _e4755_);
                        if _e4738_ {
                            phi_15739_ = true;
                        } else {
                            phi_15739_ = (_e4722_ >= _e4686_);
                        }
                        let _e4759_ = phi_15739_;
                        let _e4760_ = select(_e4686_, _e4722_, _e4759_);
                        if _e4743_ {
                            phi_15754_ = true;
                        } else {
                            phi_15754_ = (_e4727_ >= _e4691_);
                        }
                        let _e4763_ = phi_15754_;
                        let _e4764_ = select(_e4691_, _e4727_, _e4763_);
                        if _e4748_ {
                            phi_15769_ = true;
                        } else {
                            phi_15769_ = (_e4732_ >= _e4696_);
                        }
                        let _e4767_ = phi_15769_;
                        let _e4768_ = select(_e4696_, _e4732_, _e4767_);
                        if (_e4737_ != _e4737_) {
                            phi_15784_ = true;
                        } else {
                            phi_15784_ = (-1f >= _e4737_);
                        }
                        let _e4772_ = phi_15784_;
                        let _e4773_ = select(_e4737_, -1f, _e4772_);
                        if (_e4742_ != _e4742_) {
                            phi_15799_ = true;
                        } else {
                            phi_15799_ = (-1f >= _e4742_);
                        }
                        let _e4777_ = phi_15799_;
                        let _e4778_ = select(_e4742_, -1f, _e4777_);
                        if (_e4747_ != _e4747_) {
                            phi_15814_ = true;
                        } else {
                            phi_15814_ = (-1f >= _e4747_);
                        }
                        let _e4782_ = phi_15814_;
                        let _e4783_ = select(_e4747_, -1f, _e4782_);
                        if (_e4752_ != _e4752_) {
                            phi_15829_ = true;
                        } else {
                            phi_15829_ = (-1f >= _e4752_);
                        }
                        let _e4787_ = phi_15829_;
                        let _e4788_ = select(_e4752_, -1f, _e4787_);
                        if (_e4773_ != _e4773_) {
                            phi_15844_ = true;
                        } else {
                            phi_15844_ = (1f <= _e4773_);
                        }
                        let _e4792_ = phi_15844_;
                        let _e4793_ = select(_e4773_, 1f, _e4792_);
                        if (_e4778_ != _e4778_) {
                            phi_15859_ = true;
                        } else {
                            phi_15859_ = (1f <= _e4778_);
                        }
                        let _e4797_ = phi_15859_;
                        let _e4798_ = select(_e4778_, 1f, _e4797_);
                        if (_e4783_ != _e4783_) {
                            phi_15874_ = true;
                        } else {
                            phi_15874_ = (1f <= _e4783_);
                        }
                        let _e4802_ = phi_15874_;
                        let _e4803_ = select(_e4783_, 1f, _e4802_);
                        if (_e4788_ != _e4788_) {
                            phi_15889_ = true;
                        } else {
                            phi_15889_ = (1f <= _e4788_);
                        }
                        let _e4807_ = phi_15889_;
                        let _e4808_ = select(_e4788_, 1f, _e4807_);
                        if (_e4756_ != _e4756_) {
                            phi_15904_ = true;
                        } else {
                            phi_15904_ = (-1f >= _e4756_);
                        }
                        let _e4812_ = phi_15904_;
                        let _e4813_ = select(_e4756_, -1f, _e4812_);
                        if (_e4760_ != _e4760_) {
                            phi_15919_ = true;
                        } else {
                            phi_15919_ = (-1f >= _e4760_);
                        }
                        let _e4817_ = phi_15919_;
                        let _e4818_ = select(_e4760_, -1f, _e4817_);
                        if (_e4764_ != _e4764_) {
                            phi_15934_ = true;
                        } else {
                            phi_15934_ = (-1f >= _e4764_);
                        }
                        let _e4822_ = phi_15934_;
                        let _e4823_ = select(_e4764_, -1f, _e4822_);
                        if (_e4768_ != _e4768_) {
                            phi_15949_ = true;
                        } else {
                            phi_15949_ = (-1f >= _e4768_);
                        }
                        let _e4827_ = phi_15949_;
                        let _e4828_ = select(_e4768_, -1f, _e4827_);
                        if (_e4813_ != _e4813_) {
                            phi_15964_ = true;
                        } else {
                            phi_15964_ = (1f <= _e4813_);
                        }
                        let _e4832_ = phi_15964_;
                        let _e4833_ = select(_e4813_, 1f, _e4832_);
                        if (_e4818_ != _e4818_) {
                            phi_15979_ = true;
                        } else {
                            phi_15979_ = (1f <= _e4818_);
                        }
                        let _e4837_ = phi_15979_;
                        let _e4838_ = select(_e4818_, 1f, _e4837_);
                        if (_e4823_ != _e4823_) {
                            phi_15994_ = true;
                        } else {
                            phi_15994_ = (1f <= _e4823_);
                        }
                        let _e4842_ = phi_15994_;
                        let _e4843_ = select(_e4823_, 1f, _e4842_);
                        if (_e4828_ != _e4828_) {
                            phi_16009_ = true;
                        } else {
                            phi_16009_ = (1f <= _e4828_);
                        }
                        let _e4847_ = phi_16009_;
                        let _e4848_ = select(_e4828_, 1f, _e4847_);
                        let _e4853_ = (0.5f * (_e4833_ + _e4793_));
                        let _e4854_ = (0.5f * (_e4838_ + _e4798_));
                        let _e4855_ = (0.5f * (_e4843_ + _e4803_));
                        let _e4856_ = (0.5f * (_e4848_ + _e4808_));
                        let _e4857_ = (_e4833_ - _e4793_);
                        let _e4858_ = (_e4838_ - _e4798_);
                        let _e4859_ = (_e4843_ - _e4803_);
                        let _e4860_ = (_e4848_ - _e4808_);
                        let _e4861_ = (0.5f * _e4857_);
                        let _e4862_ = (0.5f * _e4858_);
                        let _e4863_ = (0.5f * _e4859_);
                        let _e4864_ = (0.5f * _e4860_);
                        let _e4865_ = (_e4853_ * _e4853_);
                        let _e4866_ = (_e4854_ * _e4854_);
                        let _e4867_ = (_e4855_ * _e4855_);
                        let _e4868_ = (_e4856_ * _e4856_);
                        let _e4869_ = (_e4861_ * _e4861_);
                        let _e4870_ = (_e4862_ * _e4862_);
                        let _e4871_ = (_e4863_ * _e4863_);
                        let _e4872_ = (_e4864_ * _e4864_);
                        if (_e4857_ != _e4857_) {
                            phi_16024_ = true;
                        } else {
                            phi_16024_ = (0f >= _e4857_);
                        }
                        let _e4876_ = phi_16024_;
                        if (_e4858_ != _e4858_) {
                            phi_16039_ = true;
                        } else {
                            phi_16039_ = (0f >= _e4858_);
                        }
                        let _e4881_ = phi_16039_;
                        if (_e4859_ != _e4859_) {
                            phi_16054_ = true;
                        } else {
                            phi_16054_ = (0f >= _e4859_);
                        }
                        let _e4886_ = phi_16054_;
                        if (_e4860_ != _e4860_) {
                            phi_16069_ = true;
                        } else {
                            phi_16069_ = (0f >= _e4860_);
                        }
                        let _e4891_ = phi_16069_;
                        let _e4893_ = (_e4756_ - _e4737_);
                        let _e4894_ = (_e4760_ - _e4742_);
                        let _e4895_ = (_e4764_ - _e4747_);
                        let _e4896_ = (_e4768_ - _e4752_);
                        let _e4899_ = bitcast<f32>((bitcast<u32>(_e4681_) & 2147483647u));
                        let _e4902_ = bitcast<f32>((bitcast<u32>(_e4686_) & 2147483647u));
                        let _e4905_ = bitcast<f32>((bitcast<u32>(_e4691_) & 2147483647u));
                        let _e4908_ = bitcast<f32>((bitcast<u32>(_e4696_) & 2147483647u));
                        let _e4911_ = bitcast<f32>((bitcast<u32>(_e4717_) & 2147483647u));
                        let _e4914_ = bitcast<f32>((bitcast<u32>(_e4722_) & 2147483647u));
                        let _e4917_ = bitcast<f32>((bitcast<u32>(_e4727_) & 2147483647u));
                        let _e4920_ = bitcast<f32>((bitcast<u32>(_e4732_) & 2147483647u));
                        if (_e4899_ != _e4899_) {
                            phi_16324_ = true;
                        } else {
                            phi_16324_ = (_e4911_ >= _e4899_);
                        }
                        let _e4924_ = phi_16324_;
                        if (_e4902_ != _e4902_) {
                            phi_16339_ = true;
                        } else {
                            phi_16339_ = (_e4914_ >= _e4902_);
                        }
                        let _e4929_ = phi_16339_;
                        if (_e4905_ != _e4905_) {
                            phi_16354_ = true;
                        } else {
                            phi_16354_ = (_e4917_ >= _e4905_);
                        }
                        let _e4934_ = phi_16354_;
                        if (_e4908_ != _e4908_) {
                            phi_16369_ = true;
                        } else {
                            phi_16369_ = (_e4920_ >= _e4908_);
                        }
                        let _e4939_ = phi_16369_;
                        let _e4945_ = (1f - (_e4865_ + _e4869_));
                        let _e4946_ = (1f - (_e4866_ + _e4870_));
                        let _e4947_ = (1f - (_e4867_ + _e4871_));
                        let _e4948_ = (1f - (_e4868_ + _e4872_));
                        let _e4974_ = select(((_e4853_ * _e4945_) * ((_e4945_ * _e4945_) + ((4f * _e4865_) * _e4869_))), 0f, (select(_e4899_, _e4911_, _e4924_) <= 0f));
                        let _e4976_ = select(((_e4854_ * _e4946_) * ((_e4946_ * _e4946_) + ((4f * _e4866_) * _e4870_))), 0f, (select(_e4902_, _e4914_, _e4929_) <= 0f));
                        let _e4978_ = select(((_e4855_ * _e4947_) * ((_e4947_ * _e4947_) + ((4f * _e4867_) * _e4871_))), 0f, (select(_e4905_, _e4917_, _e4934_) <= 0f));
                        let _e4980_ = select(((_e4856_ * _e4948_) * ((_e4948_ * _e4948_) + ((4f * _e4868_) * _e4872_))), 0f, (select(_e4908_, _e4920_, _e4939_) <= 0f));
                        if (_e4893_ != _e4893_) {
                            phi_16384_ = true;
                        } else {
                            phi_16384_ = (0.000001f >= _e4893_);
                        }
                        let _e4984_ = phi_16384_;
                        let _e4985_ = select(_e4893_, 0.000001f, _e4984_);
                        if (_e4894_ != _e4894_) {
                            phi_16399_ = true;
                        } else {
                            phi_16399_ = (0.000001f >= _e4894_);
                        }
                        let _e4989_ = phi_16399_;
                        let _e4990_ = select(_e4894_, 0.000001f, _e4989_);
                        if (_e4895_ != _e4895_) {
                            phi_16414_ = true;
                        } else {
                            phi_16414_ = (0.000001f >= _e4895_);
                        }
                        let _e4994_ = phi_16414_;
                        let _e4995_ = select(_e4895_, 0.000001f, _e4994_);
                        if (_e4896_ != _e4896_) {
                            phi_16429_ = true;
                        } else {
                            phi_16429_ = (0.000001f >= _e4896_);
                        }
                        let _e4999_ = phi_16429_;
                        let _e5000_ = select(_e4896_, 0.000001f, _e4999_);
                        let _e5003_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4985_)));
                        let _e5006_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4990_)));
                        let _e5009_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4995_)));
                        let _e5012_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5000_)));
                        let _e5022_ = (_e5003_ + (_e5003_ * (1f - (_e4985_ * _e5003_))));
                        let _e5024_ = (_e5006_ + (_e5006_ * (1f - (_e4990_ * _e5006_))));
                        let _e5026_ = (_e5009_ + (_e5009_ * (1f - (_e4995_ * _e5009_))));
                        let _e5028_ = (_e5012_ + (_e5012_ * (1f - (_e5000_ * _e5012_))));
                        let _e5105_ = (1f - _e400_);
                        let _e5106_ = (1f - _e420_);
                        let _e5107_ = (1f - _e440_);
                        let _e5108_ = (1f - _e460_);
                        phi_10218_ = type_24_(array<f32, 4>(((_e4199_.member.member[0] * _e5105_) + (_e400_ * select(select(select(_e4974_, ((select(_e4857_, 0f, _e4876_) * _e4974_) * (_e5022_ + (_e5022_ * (1f - (_e4985_ * _e5022_))))), (select((select(0f, 1f, (-1f <= _e4737_)) * select(0f, 1f, (_e4756_ <= 1f))), 1f, (_e4893_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4737_)), 0f, (_e4756_ <= -1f)))), ((_e4199_.member.member[1] * _e5106_) + (_e420_ * select(select(select(_e4976_, ((select(_e4858_, 0f, _e4881_) * _e4976_) * (_e5024_ + (_e5024_ * (1f - (_e4990_ * _e5024_))))), (select((select(0f, 1f, (-1f <= _e4742_)) * select(0f, 1f, (_e4760_ <= 1f))), 1f, (_e4894_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4742_)), 0f, (_e4760_ <= -1f)))), ((_e4199_.member.member[2] * _e5107_) + (_e440_ * select(select(select(_e4978_, ((select(_e4859_, 0f, _e4886_) * _e4978_) * (_e5026_ + (_e5026_ * (1f - (_e4995_ * _e5026_))))), (select((select(0f, 1f, (-1f <= _e4747_)) * select(0f, 1f, (_e4764_ <= 1f))), 1f, (_e4895_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4747_)), 0f, (_e4764_ <= -1f)))), ((_e4199_.member.member[3] * _e5108_) + (_e460_ * select(select(select(_e4980_, ((select(_e4860_, 0f, _e4891_) * _e4980_) * (_e5028_ + (_e5028_ * (1f - (_e5000_ * _e5028_))))), (select((select(0f, 1f, (-1f <= _e4752_)) * select(0f, 1f, (_e4768_ <= 1f))), 1f, (_e4896_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4752_)), 0f, (_e4768_ <= -1f))))));
                        phi_10219_ = type_24_(array<f32, 4>(((_e3531_.member.member[0] * _e5105_) + (_e400_ * select(select(select(_e4530_, ((select(_e4413_, 0f, _e4432_) * _e4530_) * (_e4578_ + (_e4578_ * (1f - (_e4541_ * _e4578_))))), (select((select(0f, 1f, (-1f <= _e4293_)) * select(0f, 1f, (_e4312_ <= 1f))), 1f, (_e4449_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4293_)), 0f, (_e4312_ <= -1f)))), ((_e3531_.member.member[1] * _e5106_) + (_e420_ * select(select(select(_e4532_, ((select(_e4414_, 0f, _e4437_) * _e4532_) * (_e4580_ + (_e4580_ * (1f - (_e4546_ * _e4580_))))), (select((select(0f, 1f, (-1f <= _e4298_)) * select(0f, 1f, (_e4316_ <= 1f))), 1f, (_e4450_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4298_)), 0f, (_e4316_ <= -1f)))), ((_e3531_.member.member[2] * _e5107_) + (_e440_ * select(select(select(_e4534_, ((select(_e4415_, 0f, _e4442_) * _e4534_) * (_e4582_ + (_e4582_ * (1f - (_e4551_ * _e4582_))))), (select((select(0f, 1f, (-1f <= _e4303_)) * select(0f, 1f, (_e4320_ <= 1f))), 1f, (_e4451_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4303_)), 0f, (_e4320_ <= -1f)))), ((_e3531_.member.member[3] * _e5108_) + (_e460_ * select(select(select(_e4536_, ((select(_e4416_, 0f, _e4447_) * _e4536_) * (_e4584_ + (_e4584_ * (1f - (_e4556_ * _e4584_))))), (select((select(0f, 1f, (-1f <= _e4308_)) * select(0f, 1f, (_e4324_ <= 1f))), 1f, (_e4452_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4308_)), 0f, (_e4324_ <= -1f))))));
                    }
                    let _e5154_ = phi_10218_;
                    let _e5156_ = phi_10219_;
                    if _e931_ {
                        let _e5225_ = (_e1293_ * 6u);
                        let _e5226_ = local;
                        let _e5231_ = bitcast<u32>(_e5226_[0]);
                        let _e5232_ = (_e5231_ & 2139095040u);
                        if (_e5232_ == 2139095040u) {
                            phi_10380_ = true;
                        } else {
                            phi_10380_ = (_e5232_ == 0u);
                        }
                        let _e5236_ = phi_10380_;
                        let _e5239_ = bitcast<u32>(_e5226_[1]);
                        let _e5240_ = (_e5239_ & 2139095040u);
                        if (_e5240_ == 2139095040u) {
                            phi_10396_ = true;
                        } else {
                            phi_10396_ = (_e5240_ == 0u);
                        }
                        let _e5244_ = phi_10396_;
                        let _e5247_ = bitcast<u32>(_e5226_[2]);
                        let _e5248_ = (_e5247_ & 2139095040u);
                        if (_e5248_ == 2139095040u) {
                            phi_10412_ = true;
                        } else {
                            phi_10412_ = (_e5248_ == 0u);
                        }
                        let _e5252_ = phi_10412_;
                        let _e5255_ = bitcast<u32>(_e5226_[3]);
                        let _e5256_ = (_e5255_ & 2139095040u);
                        if (_e5256_ == 2139095040u) {
                            phi_10428_ = true;
                        } else {
                            phi_10428_ = (_e5256_ == 0u);
                        }
                        let _e5260_ = phi_10428_;
                        let _e5264_ = (_e5225_ < _e113_);
                        if _e5264_ {
                            if _e5264_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5225_] = vec4<f32>(bitcast<f32>(select(_e5231_, 0u, _e5236_)), bitcast<f32>(select(_e5239_, 0u, _e5244_)), bitcast<f32>(select(_e5247_, 0u, _e5252_)), bitcast<f32>(select(_e5255_, 0u, _e5260_)));
                        }
                        let _e5267_ = (_e5225_ + 1u);
                        let _e5268_ = local_1_;
                        let _e5273_ = bitcast<u32>(_e5268_[0]);
                        let _e5274_ = (_e5273_ & 2139095040u);
                        if (_e5274_ == 2139095040u) {
                            phi_10466_ = true;
                        } else {
                            phi_10466_ = (_e5274_ == 0u);
                        }
                        let _e5278_ = phi_10466_;
                        let _e5281_ = bitcast<u32>(_e5268_[1]);
                        let _e5282_ = (_e5281_ & 2139095040u);
                        if (_e5282_ == 2139095040u) {
                            phi_10482_ = true;
                        } else {
                            phi_10482_ = (_e5282_ == 0u);
                        }
                        let _e5286_ = phi_10482_;
                        let _e5289_ = bitcast<u32>(_e5268_[2]);
                        let _e5290_ = (_e5289_ & 2139095040u);
                        if (_e5290_ == 2139095040u) {
                            phi_10498_ = true;
                        } else {
                            phi_10498_ = (_e5290_ == 0u);
                        }
                        let _e5294_ = phi_10498_;
                        let _e5297_ = bitcast<u32>(_e5268_[3]);
                        let _e5298_ = (_e5297_ & 2139095040u);
                        if (_e5298_ == 2139095040u) {
                            phi_10514_ = true;
                        } else {
                            phi_10514_ = (_e5298_ == 0u);
                        }
                        let _e5302_ = phi_10514_;
                        let _e5306_ = (_e5267_ < _e113_);
                        if _e5306_ {
                            if _e5306_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5267_] = vec4<f32>(bitcast<f32>(select(_e5273_, 0u, _e5278_)), bitcast<f32>(select(_e5281_, 0u, _e5286_)), bitcast<f32>(select(_e5289_, 0u, _e5294_)), bitcast<f32>(select(_e5297_, 0u, _e5302_)));
                        }
                        let _e5309_ = (_e5225_ + 2u);
                        let _e5310_ = bitcast<u32>(_e5156_.member[0]);
                        let _e5311_ = (_e5310_ & 2139095040u);
                        if (_e5311_ == 2139095040u) {
                            phi_10546_ = true;
                        } else {
                            phi_10546_ = (_e5311_ == 0u);
                        }
                        let _e5315_ = phi_10546_;
                        let _e5318_ = bitcast<u32>(_e5156_.member[1]);
                        let _e5319_ = (_e5318_ & 2139095040u);
                        if (_e5319_ == 2139095040u) {
                            phi_10562_ = true;
                        } else {
                            phi_10562_ = (_e5319_ == 0u);
                        }
                        let _e5323_ = phi_10562_;
                        let _e5326_ = bitcast<u32>(_e5156_.member[2]);
                        let _e5327_ = (_e5326_ & 2139095040u);
                        if (_e5327_ == 2139095040u) {
                            phi_10578_ = true;
                        } else {
                            phi_10578_ = (_e5327_ == 0u);
                        }
                        let _e5331_ = phi_10578_;
                        let _e5334_ = bitcast<u32>(_e5156_.member[3]);
                        let _e5335_ = (_e5334_ & 2139095040u);
                        if (_e5335_ == 2139095040u) {
                            phi_10594_ = true;
                        } else {
                            phi_10594_ = (_e5335_ == 0u);
                        }
                        let _e5339_ = phi_10594_;
                        let _e5343_ = (_e5309_ < _e113_);
                        if _e5343_ {
                            if _e5343_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5309_] = vec4<f32>(bitcast<f32>(select(_e5310_, 0u, _e5315_)), bitcast<f32>(select(_e5318_, 0u, _e5323_)), bitcast<f32>(select(_e5326_, 0u, _e5331_)), bitcast<f32>(select(_e5334_, 0u, _e5339_)));
                        }
                        let _e5346_ = (_e5225_ + 3u);
                        let _e5347_ = bitcast<u32>(_e5154_.member[0]);
                        let _e5348_ = (_e5347_ & 2139095040u);
                        if (_e5348_ == 2139095040u) {
                            phi_10626_ = true;
                        } else {
                            phi_10626_ = (_e5348_ == 0u);
                        }
                        let _e5352_ = phi_10626_;
                        let _e5355_ = bitcast<u32>(_e5154_.member[1]);
                        let _e5356_ = (_e5355_ & 2139095040u);
                        if (_e5356_ == 2139095040u) {
                            phi_10642_ = true;
                        } else {
                            phi_10642_ = (_e5356_ == 0u);
                        }
                        let _e5360_ = phi_10642_;
                        let _e5363_ = bitcast<u32>(_e5154_.member[2]);
                        let _e5364_ = (_e5363_ & 2139095040u);
                        if (_e5364_ == 2139095040u) {
                            phi_10658_ = true;
                        } else {
                            phi_10658_ = (_e5364_ == 0u);
                        }
                        let _e5368_ = phi_10658_;
                        let _e5371_ = bitcast<u32>(_e5154_.member[3]);
                        let _e5372_ = (_e5371_ & 2139095040u);
                        if (_e5372_ == 2139095040u) {
                            phi_10674_ = true;
                        } else {
                            phi_10674_ = (_e5372_ == 0u);
                        }
                        let _e5376_ = phi_10674_;
                        let _e5380_ = (_e5346_ < _e113_);
                        if _e5380_ {
                            if _e5380_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5346_] = vec4<f32>(bitcast<f32>(select(_e5347_, 0u, _e5352_)), bitcast<f32>(select(_e5355_, 0u, _e5360_)), bitcast<f32>(select(_e5363_, 0u, _e5368_)), bitcast<f32>(select(_e5371_, 0u, _e5376_)));
                        }
                        let _e5383_ = (_e5225_ + 4u);
                        let _e5384_ = bitcast<u32>((_e3531_.member_1_.member[0] + (_e378_.x * (_e5156_.member[0] * _e5156_.member[1]))));
                        let _e5385_ = (_e5384_ & 2139095040u);
                        if (_e5385_ == 2139095040u) {
                            phi_10706_ = true;
                        } else {
                            phi_10706_ = (_e5385_ == 0u);
                        }
                        let _e5389_ = phi_10706_;
                        let _e5392_ = bitcast<u32>((_e3531_.member_1_.member[1] + (_e378_.y * (_e5156_.member[1] * _e5156_.member[2]))));
                        let _e5393_ = (_e5392_ & 2139095040u);
                        if (_e5393_ == 2139095040u) {
                            phi_10722_ = true;
                        } else {
                            phi_10722_ = (_e5393_ == 0u);
                        }
                        let _e5397_ = phi_10722_;
                        let _e5400_ = bitcast<u32>((_e3531_.member_1_.member[2] + (_e378_.z * (_e5156_.member[2] * _e5156_.member[3]))));
                        let _e5401_ = (_e5400_ & 2139095040u);
                        if (_e5401_ == 2139095040u) {
                            phi_10738_ = true;
                        } else {
                            phi_10738_ = (_e5401_ == 0u);
                        }
                        let _e5405_ = phi_10738_;
                        let _e5408_ = bitcast<u32>((_e3531_.member_1_.member[3] + (_e378_.w * (_e5156_.member[3] * _e5156_.member[0]))));
                        let _e5409_ = (_e5408_ & 2139095040u);
                        if (_e5409_ == 2139095040u) {
                            phi_10754_ = true;
                        } else {
                            phi_10754_ = (_e5409_ == 0u);
                        }
                        let _e5413_ = phi_10754_;
                        let _e5417_ = (_e5383_ < _e113_);
                        if _e5417_ {
                            if _e5417_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5383_] = vec4<f32>(bitcast<f32>(select(_e5384_, 0u, _e5389_)), bitcast<f32>(select(_e5392_, 0u, _e5397_)), bitcast<f32>(select(_e5400_, 0u, _e5405_)), bitcast<f32>(select(_e5408_, 0u, _e5413_)));
                        }
                        let _e5420_ = (_e5225_ + 5u);
                        let _e5421_ = bitcast<u32>((_e4199_.member_1_.member[0] + (_e378_.x * (_e5154_.member[0] * _e5154_.member[1]))));
                        let _e5422_ = (_e5421_ & 2139095040u);
                        if (_e5422_ == 2139095040u) {
                            phi_10786_ = true;
                        } else {
                            phi_10786_ = (_e5422_ == 0u);
                        }
                        let _e5426_ = phi_10786_;
                        let _e5429_ = bitcast<u32>((_e4199_.member_1_.member[1] + (_e378_.y * (_e5154_.member[1] * _e5154_.member[2]))));
                        let _e5430_ = (_e5429_ & 2139095040u);
                        if (_e5430_ == 2139095040u) {
                            phi_10802_ = true;
                        } else {
                            phi_10802_ = (_e5430_ == 0u);
                        }
                        let _e5434_ = phi_10802_;
                        let _e5437_ = bitcast<u32>((_e4199_.member_1_.member[2] + (_e378_.z * (_e5154_.member[2] * _e5154_.member[3]))));
                        let _e5438_ = (_e5437_ & 2139095040u);
                        if (_e5438_ == 2139095040u) {
                            phi_10818_ = true;
                        } else {
                            phi_10818_ = (_e5438_ == 0u);
                        }
                        let _e5442_ = phi_10818_;
                        let _e5445_ = bitcast<u32>((_e4199_.member_1_.member[3] + (_e378_.w * (_e5154_.member[3] * _e5154_.member[0]))));
                        let _e5446_ = (_e5445_ & 2139095040u);
                        if (_e5446_ == 2139095040u) {
                            phi_10834_ = true;
                        } else {
                            phi_10834_ = (_e5446_ == 0u);
                        }
                        let _e5450_ = phi_10834_;
                        let _e5454_ = (_e5420_ < _e113_);
                        if _e5454_ {
                            if _e5454_ {
                            } else {
                                phi_17053_ = true;
                                break;
                            }
                            feature_outputs.member[_e5420_] = vec4<f32>(bitcast<f32>(select(_e5421_, 0u, _e5426_)), bitcast<f32>(select(_e5429_, 0u, _e5434_)), bitcast<f32>(select(_e5437_, 0u, _e5442_)), bitcast<f32>(select(_e5445_, 0u, _e5450_)));
                        }
                    }
                    if (_e929_ > 0u) {
                        let _e5459_ = ((_e929_ - 1u) & 255u);
                        if (_e5459_ < 512u) {
                        } else {
                            phi_17053_ = true;
                            break;
                        }
                        let _e5462_ = lane_buf[_e5459_];
                        let _e5464_ = bitcast<u32>(_e5462_.x);
                        let _e5465_ = (_e5464_ & 2139095040u);
                        if (_e5465_ == 2139095040u) {
                            phi_10875_ = true;
                        } else {
                            phi_10875_ = (_e5465_ == 0u);
                        }
                        let _e5469_ = phi_10875_;
                        let _e5471_ = bitcast<f32>(select(_e5464_, 0u, _e5469_));
                        if (_e5471_ < -1000000000000000000f) {
                            phi_10892_ = -1000000000000000000f;
                        } else {
                            phi_10892_ = select(_e5471_, 1000000000000000000f, (_e5471_ > 1000000000000000000f));
                        }
                        let _e5476_ = phi_10892_;
                        let _e5478_ = bitcast<u32>(_e5462_.y);
                        let _e5479_ = (_e5478_ & 2139095040u);
                        if (_e5479_ == 2139095040u) {
                            phi_10904_ = true;
                        } else {
                            phi_10904_ = (_e5479_ == 0u);
                        }
                        let _e5483_ = phi_10904_;
                        let _e5485_ = bitcast<f32>(select(_e5478_, 0u, _e5483_));
                        if (_e5485_ < -1000000000000000000f) {
                            phi_10921_ = -1000000000000000000f;
                        } else {
                            phi_10921_ = select(_e5485_, 1000000000000000000f, (_e5485_ > 1000000000000000000f));
                        }
                        let _e5490_ = phi_10921_;
                        let _e5492_ = bitcast<u32>(_e5462_.z);
                        let _e5493_ = (_e5492_ & 2139095040u);
                        if (_e5493_ == 2139095040u) {
                            phi_10933_ = true;
                        } else {
                            phi_10933_ = (_e5493_ == 0u);
                        }
                        let _e5497_ = phi_10933_;
                        let _e5499_ = bitcast<f32>(select(_e5492_, 0u, _e5497_));
                        if (_e5499_ < -1000000000000000000f) {
                            phi_10950_ = -1000000000000000000f;
                        } else {
                            phi_10950_ = select(_e5499_, 1000000000000000000f, (_e5499_ > 1000000000000000000f));
                        }
                        let _e5504_ = phi_10950_;
                        let _e5506_ = bitcast<u32>(_e5462_.w);
                        let _e5507_ = (_e5506_ & 2139095040u);
                        if (_e5507_ == 2139095040u) {
                            phi_10962_ = true;
                        } else {
                            phi_10962_ = (_e5507_ == 0u);
                        }
                        let _e5511_ = phi_10962_;
                        let _e5513_ = bitcast<f32>(select(_e5506_, 0u, _e5511_));
                        if (_e5513_ < -1000000000000000000f) {
                            phi_10979_ = -1000000000000000000f;
                        } else {
                            phi_10979_ = select(_e5513_, 1000000000000000000f, (_e5513_ > 1000000000000000000f));
                        }
                        let _e5518_ = phi_10979_;
                        let _e5520_ = (256u + _e5459_);
                        if (_e5520_ < 512u) {
                        } else {
                            phi_17053_ = true;
                            break;
                        }
                        let _e5523_ = lane_buf[_e5520_];
                        let _e5525_ = bitcast<u32>(_e5523_.x);
                        let _e5526_ = (_e5525_ & 2139095040u);
                        if (_e5526_ == 2139095040u) {
                            phi_11000_ = true;
                        } else {
                            phi_11000_ = (_e5526_ == 0u);
                        }
                        let _e5530_ = phi_11000_;
                        let _e5532_ = bitcast<f32>(select(_e5525_, 0u, _e5530_));
                        if (_e5532_ < -1000000000000000000f) {
                            phi_11017_ = -1000000000000000000f;
                        } else {
                            phi_11017_ = select(_e5532_, 1000000000000000000f, (_e5532_ > 1000000000000000000f));
                        }
                        let _e5537_ = phi_11017_;
                        let _e5539_ = bitcast<u32>(_e5523_.y);
                        let _e5540_ = (_e5539_ & 2139095040u);
                        if (_e5540_ == 2139095040u) {
                            phi_11029_ = true;
                        } else {
                            phi_11029_ = (_e5540_ == 0u);
                        }
                        let _e5544_ = phi_11029_;
                        let _e5546_ = bitcast<f32>(select(_e5539_, 0u, _e5544_));
                        if (_e5546_ < -1000000000000000000f) {
                            phi_11046_ = -1000000000000000000f;
                        } else {
                            phi_11046_ = select(_e5546_, 1000000000000000000f, (_e5546_ > 1000000000000000000f));
                        }
                        let _e5551_ = phi_11046_;
                        let _e5553_ = bitcast<u32>(_e5523_.z);
                        let _e5554_ = (_e5553_ & 2139095040u);
                        if (_e5554_ == 2139095040u) {
                            phi_11058_ = true;
                        } else {
                            phi_11058_ = (_e5554_ == 0u);
                        }
                        let _e5558_ = phi_11058_;
                        let _e5560_ = bitcast<f32>(select(_e5553_, 0u, _e5558_));
                        if (_e5560_ < -1000000000000000000f) {
                            phi_11075_ = -1000000000000000000f;
                        } else {
                            phi_11075_ = select(_e5560_, 1000000000000000000f, (_e5560_ > 1000000000000000000f));
                        }
                        let _e5565_ = phi_11075_;
                        let _e5567_ = bitcast<u32>(_e5523_.w);
                        let _e5568_ = (_e5567_ & 2139095040u);
                        if (_e5568_ == 2139095040u) {
                            phi_11087_ = true;
                        } else {
                            phi_11087_ = (_e5568_ == 0u);
                        }
                        let _e5572_ = phi_11087_;
                        let _e5574_ = bitcast<f32>(select(_e5567_, 0u, _e5572_));
                        if (_e5574_ < -1000000000000000000f) {
                            phi_11104_ = -1000000000000000000f;
                        } else {
                            phi_11104_ = select(_e5574_, 1000000000000000000f, (_e5574_ > 1000000000000000000f));
                        }
                        let _e5579_ = phi_11104_;
                        phi_11109_ = type_32_(_e2824_.member, _e2824_.member_1_, _e2824_.member_2_, type_27_(type_24_(array<f32, 4>(_e5476_, _e5490_, _e5504_, _e5518_)), type_24_(array<f32, 4>(_e5537_, _e5551_, _e5565_, _e5579_))));
                    } else {
                        phi_11109_ = _e2824_;
                    }
                    let _e5590_ = phi_11109_;
                    workgroupBarrier();
                    phi_17054_ = _e2622_;
                    phi_1622_ = _e5590_;
                } else {
                    phi_17054_ = _e899_;
                    phi_1622_ = type_32_();
                }
                let _e5592_ = phi_17054_;
                let _e5594_ = phi_1622_;
                phi_16480_ = _e5592_;
                phi_1618_ = _e916_;
                phi_1621_ = _e5594_;
                phi_17053_ = _e5592_;
                if !(_e922_) {
                    break;
                }
            }
            let _e5597_ = phi_17053_;
            if _e5597_ {
                break;
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn fused_core_scan_256(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    num_wg_1_ = num_wg;
    lid_1_ = lid;
    function_();
    return;
}
